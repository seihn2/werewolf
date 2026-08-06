from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .llm import ChatBackend
from .models import (
    ActionType,
    AgentObservation,
    CandidateAction,
    CandidateEvaluation,
    DecisionTrace,
    GameAction,
    Phase,
    Role,
)


class Planner(Protocol):
    async def plan(self, observation: AgentObservation) -> list[CandidateAction]: ...


class Evaluator(Protocol):
    async def evaluate(
        self, observation: AgentObservation, candidate: CandidateAction
    ) -> CandidateEvaluation: ...


class Executor(Protocol):
    def execute(self, observation: AgentObservation, candidate: CandidateAction) -> GameAction: ...


class Reflexion(Protocol):
    async def revise(
        self,
        observation: AgentObservation,
        candidate: CandidateAction,
        evaluation: CandidateEvaluation,
    ) -> tuple[CandidateAction, str]: ...


def expected_action_type(phase: Phase) -> ActionType:
    mapping = {
        Phase.NIGHT_WEREWOLF: ActionType.KILL,
        Phase.NIGHT_SEER: ActionType.CHECK,
        Phase.NIGHT_DOCTOR: ActionType.PROTECT,
        Phase.DAY_DISCUSSION: ActionType.SPEAK,
        Phase.DAY_VOTE: ActionType.VOTE,
    }
    try:
        return mapping[phase]
    except KeyError as error:
        raise ValueError(f"phase does not accept agent decisions: {phase}") from error


def _legal_targets(observation: AgentObservation) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                target_id.strip()
                for target_id in observation.legal_targets
                if isinstance(target_id, str) and target_id.strip()
            }
        )
    )


def _role_probability(observation: AgentObservation, target_id: str, role: Role) -> float:
    raw_value = observation.role_beliefs.get(target_id, {}).get(role.value, 0.0)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _has_role_belief(observation: AgentObservation, target_id: str, role: Role) -> bool:
    return role.value in observation.role_beliefs.get(target_id, {})


def _wolf_probability(observation: AgentObservation, target_id: str) -> float:
    return _role_probability(observation, target_id, Role.WEREWOLF)


def _claims_seer(message: str) -> bool:
    normalized = message.casefold().replace(" ", "")
    negative_markers = ("我不是预言家", "并非预言家", "nottheseer")
    if any(marker in normalized for marker in negative_markers):
        return False
    claim_markers = (
        "我是预言家",
        "我跳预言家",
        "我起跳预言家",
        "预言家起跳",
        "预言家报验",
        "iamtheseer",
        "i'mtheseer",
    )
    return any(marker in normalized for marker in claim_markers)


def _seer_claimants(observation: AgentObservation) -> tuple[str, ...]:
    legal = set(_legal_targets(observation))
    claimants: list[str] = []
    seen: set[str] = set()
    for event in reversed(observation.events):
        sender = event.sender
        if event.topic != "speech" or sender not in legal or sender in seen:
            continue
        message = str(event.payload.get("message", ""))
        if _claims_seer(message):
            claimants.append(sender)
            seen.add(sender)
    return tuple(claimants)


def _message_mentions(message: str, target_id: str) -> bool:
    return bool(target_id) and target_id.casefold() in message.casefold()


def _public_pressure(observation: AgentObservation, target_id: str) -> float:
    pressure = 0.0
    pressure_words = ("投", "票", "出", "狼人", "怀疑", "vote", "wolf")
    for event in observation.events:
        if event.topic == "speech":
            message = str(event.payload.get("message", ""))
            if _message_mentions(message, target_id):
                pressure += 1.0
                if any(word in message.casefold() for word in pressure_words):
                    pressure += 0.5
        elif event.topic == "vote_cast" and event.payload.get("target_id") == target_id:
            pressure += 1.5
        elif event.topic == "vote_result":
            tally = event.payload.get("tally", {})
            if isinstance(tally, Mapping):
                try:
                    pressure += 0.5 * float(tally.get(target_id, 0))
                except (TypeError, ValueError):
                    continue
    return pressure


def _speaker_influence(observation: AgentObservation, target_id: str) -> float:
    speeches = sum(
        1 for event in observation.events if event.topic == "speech" and event.sender == target_id
    )
    return float(speeches) + (1.0 if target_id in _seer_claimants(observation) else 0.0)


def _latest_declared_target(
    observation: AgentObservation, *, sender: str | None = None
) -> str | None:
    legal_targets = _legal_targets(observation)
    for event in reversed(observation.events):
        if event.topic != "speech" or (sender is not None and event.sender != sender):
            continue
        message = str(event.payload.get("message", ""))
        mentioned = [
            target_id for target_id in legal_targets if _message_mentions(message, target_id)
        ]
        if mentioned:
            return min(mentioned, key=lambda target_id: (message.find(target_id), target_id))
    return None


def _last_protected_target(observation: AgentObservation) -> str | None:
    for event in reversed(observation.events):
        if event.topic == "doctor_choice":
            target_id = event.payload.get("target_id")
            return target_id if isinstance(target_id, str) else None
    return None


def _known_wolves(observation: AgentObservation) -> tuple[str, ...]:
    return tuple(
        target_id
        for target_id in _legal_targets(observation)
        if _has_role_belief(observation, target_id, Role.WEREWOLF)
        and _wolf_probability(observation, target_id) >= 0.99
    )


def _known_villagers(observation: AgentObservation) -> tuple[str, ...]:
    return tuple(
        target_id
        for target_id in _legal_targets(observation)
        if _has_role_belief(observation, target_id, Role.WEREWOLF)
        and _wolf_probability(observation, target_id) <= 0.01
    )


def _candidate_signature(candidate: CandidateAction) -> tuple[ActionType, str | None, str]:
    return candidate.action_type, candidate.target_id, candidate.message.strip()


def _legality_reasons(
    observation: AgentObservation,
    candidate: CandidateAction,
    expected: ActionType | None = None,
) -> tuple[str, ...]:
    expected = expected or expected_action_type(observation.phase)
    legal_targets = _legal_targets(observation)

    if candidate.action_type is ActionType.ABSTAIN:
        reasons: list[str] = []
        if candidate.target_id is not None:
            reasons.append("Abstain actions must not include a target.")
        if expected is not ActionType.VOTE and (expected is ActionType.SPEAK or legal_targets):
            reasons.append("Abstain is only allowed for voting or when no target exists.")
        return tuple(reasons)

    if candidate.action_type is not expected:
        return (f"Expected action type '{expected.value}', got '{candidate.action_type.value}'.",)

    if expected is ActionType.SPEAK:
        reasons = []
        if candidate.target_id is not None:
            reasons.append("Speech actions must not include a target.")
        if not candidate.message.strip():
            reasons.append("Speech is empty.")
        return tuple(reasons)

    if candidate.target_id not in legal_targets:
        return (f"Target '{candidate.target_id}' is not legal in the current phase.",)
    return ()


def _candidate_is_legal(
    observation: AgentObservation,
    candidate: CandidateAction,
    expected: ActionType | None = None,
) -> bool:
    return not _legality_reasons(observation, candidate, expected)


def _stable_integer(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(raw.encode()).hexdigest()[:16], 16)


class HeuristicPlanner:
    def __init__(self, seed: int) -> None:
        self._seed = seed

    async def plan(self, observation: AgentObservation) -> list[CandidateAction]:
        expected = expected_action_type(observation.phase)
        if expected is ActionType.SPEAK:
            return self._discussion_candidates(observation)

        targets = _legal_targets(observation)
        if not targets:
            return [
                CandidateAction(
                    ActionType.ABSTAIN,
                    "no_legal_target",
                    rationale="No legal target exists, so abstaining is the only safe action.",
                )
            ]
        if expected is ActionType.KILL and observation.role is Role.WEREWOLF:
            return self._werewolf_kill_candidates(observation)
        if expected is ActionType.CHECK and observation.role is Role.SEER:
            return self._seer_check_candidates(observation)
        if expected is ActionType.PROTECT and observation.role is Role.DOCTOR:
            return self._doctor_protect_candidates(observation)
        if expected is ActionType.VOTE:
            return self._vote_candidates(observation)
        return self._generic_target_candidates(observation, expected)

    def _sort_targets(
        self, observation: AgentObservation, scores: Mapping[str, float]
    ) -> list[str]:
        return sorted(
            scores,
            key=lambda target_id: (
                -scores[target_id],
                _stable_integer(
                    self._seed,
                    observation.game_id,
                    observation.round_no,
                    observation.phase.value,
                    observation.player_id,
                    target_id,
                ),
                target_id,
            ),
        )

    def _werewolf_kill_candidates(self, observation: AgentObservation) -> list[CandidateAction]:
        claimants = set(_seer_claimants(observation))
        scores = {
            target_id: (
                1.6 * (target_id in claimants)
                + 0.25 * _speaker_influence(observation, target_id)
                + 0.12 * _public_pressure(observation, target_id)
                + 0.2 * (1.0 - _wolf_probability(observation, target_id))
            )
            for target_id in _legal_targets(observation)
            if target_id not in observation.teammate_ids
        }
        ranked = self._sort_targets(observation, scores)
        candidates = []
        for index, target_id in enumerate(ranked[:3]):
            if target_id in claimants:
                strategy = "kill_seer_claimant"
                rationale = "Remove a public Seer claimant before another check is reported."
            elif index == 0:
                strategy = "kill_influencer"
                rationale = "Remove the most influential non-teammate village voice."
            else:
                strategy = "night_kill_priority"
                rationale = "Choose a legal non-teammate target with strategic value."
            candidates.append(
                CandidateAction(
                    ActionType.KILL,
                    strategy,
                    target_id=target_id,
                    rationale=rationale,
                )
            )
        return candidates

    def _seer_check_candidates(self, observation: AgentObservation) -> list[CandidateAction]:
        claimants = set(_seer_claimants(observation))
        scores: dict[str, float] = {}
        for target_id in _legal_targets(observation):
            wolf_probability = _wolf_probability(observation, target_id)
            has_belief = _has_role_belief(observation, target_id, Role.WEREWOLF)
            confirmed = has_belief and (wolf_probability <= 0.01 or wolf_probability >= 0.99)
            uncertainty = (
                1.0 if not has_belief else max(0.0, 1.0 - abs(wolf_probability - 0.5) * 2.0)
            )
            scores[target_id] = (
                (0.0 if confirmed else 1.0)
                + 0.5 * uncertainty
                + 0.35 * wolf_probability
                + 0.45 * (target_id in claimants)
                + 0.08 * _speaker_influence(observation, target_id)
            )

        ranked = self._sort_targets(observation, scores)
        candidates = []
        for target_id in ranked[:3]:
            wolf_probability = _wolf_probability(observation, target_id)
            if target_id in claimants:
                strategy = "check_counterclaim"
                rationale = "Verify a public Seer claimant before trusting their reports."
            elif wolf_probability >= 0.55:
                strategy = "check_suspect"
                rationale = "Resolve a high-risk but unconfirmed suspect."
            elif not _has_role_belief(observation, target_id, Role.WEREWOLF):
                strategy = "check_unknown"
                rationale = "Maximize information gain by checking an unknown player."
            else:
                strategy = "check_information_gain"
                rationale = "Prefer uncertain alignments over repeating a confirmed check."
            candidates.append(
                CandidateAction(
                    ActionType.CHECK,
                    strategy,
                    target_id=target_id,
                    rationale=rationale,
                )
            )
        return candidates

    def _doctor_protect_candidates(self, observation: AgentObservation) -> list[CandidateAction]:
        claimants = set(_seer_claimants(observation))
        previous_target = _last_protected_target(observation)
        scores = {
            target_id: (
                2.0 * (target_id in claimants)
                + 0.25 * _speaker_influence(observation, target_id)
                + 0.35 * (target_id == observation.player_id)
                - 0.9 * _wolf_probability(observation, target_id)
                - 0.12 * (target_id == previous_target)
            )
            for target_id in _legal_targets(observation)
        }
        ranked = self._sort_targets(observation, scores)
        candidates = []
        for target_id in ranked[:3]:
            if target_id in claimants:
                strategy = "protect_seer_claim"
                rationale = "Protect the public information role most likely to be attacked."
            elif target_id == observation.player_id:
                strategy = "self_preservation"
                rationale = "Preserve the Doctor when no stronger public information role exists."
            else:
                strategy = "protect_influencer"
                rationale = "Protect an influential village voice while avoiding obvious wolves."
            candidates.append(
                CandidateAction(
                    ActionType.PROTECT,
                    strategy,
                    target_id=target_id,
                    rationale=rationale,
                )
            )
        return candidates

    def _vote_candidates(self, observation: AgentObservation) -> list[CandidateAction]:
        targets = list(_legal_targets(observation))
        if observation.role is Role.WEREWOLF:
            non_teammates = [
                target_id for target_id in targets if target_id not in observation.teammate_ids
            ]
            if non_teammates:
                targets = non_teammates
        declared_target = _latest_declared_target(observation, sender=observation.player_id)
        known_wolves = set(_known_wolves(observation))
        scores = {
            target_id: (
                1.8 * (target_id in known_wolves and observation.role is not Role.WEREWOLF)
                + 1.4 * (target_id == declared_target)
                + 0.35 * _public_pressure(observation, target_id)
                + 0.55 * _wolf_probability(observation, target_id)
            )
            for target_id in targets
        }
        if observation.role is Role.WEREWOLF:
            scores = {
                target_id: score + 0.25 * (1.0 - _wolf_probability(observation, target_id))
                for target_id, score in scores.items()
            }
        ranked = self._sort_targets(observation, scores)
        candidates = []
        for index, target_id in enumerate(ranked[:3]):
            if observation.role is Role.WEREWOLF and target_id == declared_target:
                strategy = "follow_through_vote"
                rationale = (
                    "Follow through on the public push to keep the fabricated story coherent."
                )
            elif observation.role is Role.WEREWOLF and index == 0:
                strategy = "lead_vote"
                rationale = "Lead a consolidated wagon against a non-teammate target."
            elif target_id in known_wolves:
                strategy = "vote_known_wolf"
                rationale = "Convert confirmed Seer information into a decisive vote."
            elif _public_pressure(observation, target_id) > 0:
                strategy = "evidence_vote"
                rationale = (
                    "Vote where public claims and prior pressure provide accountable evidence."
                )
            else:
                strategy = "vote_suspect"
                rationale = "Vote the highest-ranked legal suspect."
            candidates.append(
                CandidateAction(
                    ActionType.VOTE,
                    strategy,
                    target_id=target_id,
                    rationale=rationale,
                )
            )
        candidates.append(
            CandidateAction(
                ActionType.ABSTAIN,
                "abstain",
                rationale="Keep abstention as a legal but strategically weak fallback.",
            )
        )
        return candidates

    def _generic_target_candidates(
        self, observation: AgentObservation, expected: ActionType
    ) -> list[CandidateAction]:
        scores = {
            target_id: (
                _wolf_probability(observation, target_id)
                + 0.1 * _public_pressure(observation, target_id)
            )
            for target_id in _legal_targets(observation)
        }
        ranked = self._sort_targets(observation, scores)
        return [
            CandidateAction(
                action_type=expected,
                strategy=f"target_priority_{index + 1}",
                target_id=target_id,
                rationale="Choose a legal target using deterministic strategic ranking.",
            )
            for index, target_id in enumerate(ranked[:3])
        ]

    def _discussion_candidates(self, observation: AgentObservation) -> list[CandidateAction]:
        if observation.role is Role.WEREWOLF:
            return self._werewolf_discussion_candidates(observation)
        if observation.role is Role.SEER:
            return self._seer_discussion_candidates(observation)
        if observation.role is Role.DOCTOR:
            return self._doctor_discussion_candidates(observation)
        return self._village_discussion_candidates(observation)

    def _rank_discussion_targets(self, observation: AgentObservation) -> list[str]:
        targets = [
            target_id
            for target_id in _legal_targets(observation)
            if target_id not in observation.teammate_ids
        ]
        scores = {
            target_id: (
                1.4 * _wolf_probability(observation, target_id)
                + 0.35 * _public_pressure(observation, target_id)
                + 0.1 * _speaker_influence(observation, target_id)
            )
            for target_id in targets
        }
        if observation.role is Role.WEREWOLF:
            scores = {
                target_id: score + 0.2 * (1.0 - _wolf_probability(observation, target_id))
                for target_id, score in scores.items()
            }
        return self._sort_targets(observation, scores)

    def _werewolf_discussion_candidates(
        self, observation: AgentObservation
    ) -> list[CandidateAction]:
        targets = self._rank_discussion_targets(observation)
        primary = targets[0] if targets else "当前票型焦点"
        secondary = targets[1] if len(targets) > 1 else primary
        return [
            CandidateAction(
                ActionType.SPEAK,
                "fake_seer_claim",
                message=(
                    f"我起跳预言家：昨晚查验 {primary} 是狼人。今天票归 {primary}，"
                    f"好人先统一票型；如果需要留警徽流，我下一轮查 {secondary}。"
                ),
                rationale="Use a coherent aggressive fake-Seer claim (悍跳) with a concrete vote.",
            ),
            CandidateAction(
                ActionType.SPEAK,
                "lead_vote",
                message=(
                    f"不要继续分散票型，{primary} 的发言与公开信息冲突最大。"
                    f"我今天明确带票 {primary}，其他人请给出跟票或反对的可验证理由。"
                ),
                rationale="Lead the vote (带票) without exposing the Werewolf team.",
            ),
            CandidateAction(
                ActionType.SPEAK,
                "conceal_role",
                message=(
                    f"我先站边公开信息：目前 {primary} 的嫌疑最高，但请大家继续核对"
                    "前后发言和历史票型，避免狼人利用情绪分票。"
                ),
                rationale="Blend into the village while preserving a consistent public position.",
            ),
        ]

    def _seer_discussion_candidates(self, observation: AgentObservation) -> list[CandidateAction]:
        known_wolves = _known_wolves(observation)
        targets = self._rank_discussion_targets(observation)
        if known_wolves:
            known_wolf = known_wolves[0]
            counterclaim = known_wolf in _seer_claimants(observation)
            return [
                CandidateAction(
                    ActionType.SPEAK,
                    "reveal_and_accuse",
                    message=(
                        f"我是预言家，昨晚查验 {known_wolf} 是狼人。"
                        f"今天必须统一投 {known_wolf}，这是确定验人信息。"
                    ),
                    rationale="Reveal decisive private information without inventing extra checks.",
                ),
                CandidateAction(
                    ActionType.SPEAK,
                    "coordinate_vote",
                    message=(
                        f"我的验人结论是 {known_wolf} 为狼人。好人票归 {known_wolf}，"
                        "任何试图分票的人都需要解释其收益。"
                    ),
                    rationale="Turn a confirmed check into coordinated village action.",
                ),
                CandidateAction(
                    ActionType.SPEAK,
                    "counter_fake_seer" if counterclaim else "evidence_chain",
                    message=(
                        f"请复盘 {known_wolf} 的身份声明、发言和票型；我的查验已经确认他是狼人，"
                        "不要被反向悍跳或临时改口带偏。"
                    ),
                    rationale="Defend the genuine Seer result against misinformation.",
                ),
            ]

        known_villagers = _known_villagers(observation)
        target = targets[0] if targets else "当前高风险玩家"
        candidates = [
            CandidateAction(
                ActionType.SPEAK,
                "information_request",
                message="每个人请明确一名怀疑对象、证据和预期票型，避免只报结论不报逻辑。",
                rationale="Collect falsifiable public information before revealing the role.",
            ),
            CandidateAction(
                ActionType.SPEAK,
                "soft_guidance",
                message=f"我当前重点观察 {target}，请他解释发言变化和投票收益。",
                rationale="Guide discussion without fabricating a check result.",
            ),
            CandidateAction(
                ActionType.SPEAK,
                "vote_coordination",
                message=f"临近投票不要散票，暂时围绕 {target} 对齐证据，再决定最终票型。",
                rationale="Reduce village coordination failure while preserving flexibility.",
            ),
        ]
        if known_villagers:
            good_player = known_villagers[0]
            candidates[0] = CandidateAction(
                ActionType.SPEAK,
                "share_golden_water",
                message=f"我有可靠信息确认 {good_player} 不是狼人，今天不要把票浪费在他身上。",
                rationale="Share a genuine non-wolf check without inventing additional certainty.",
            )
        return candidates

    def _doctor_discussion_candidates(self, observation: AgentObservation) -> list[CandidateAction]:
        targets = self._rank_discussion_targets(observation)
        target = targets[0] if targets else "当前高风险玩家"
        claimants = _seer_claimants(observation)
        information_role = claimants[0] if claimants else None
        return [
            CandidateAction(
                ActionType.SPEAK,
                "conceal_doctor",
                message=f"我更怀疑 {target}，请他把身份判断、行为收益和最终票型说完整。",
                rationale="Contribute useful analysis without exposing the Doctor role.",
            ),
            CandidateAction(
                ActionType.SPEAK,
                "protect_information_flow",
                message=(
                    f"先核验 {information_role} 的验人链和后续承诺，不要急着分散火力。"
                    if information_role
                    else "请信息位给出可验证的结果和后续计划，其他人不要抢先暴露底牌。"
                ),
                rationale="Protect information quality while keeping the Doctor hidden.",
            ),
            CandidateAction(
                ActionType.SPEAK,
                "vote_coordination",
                message=f"今天先围绕 {target} 汇总证据，投票前明确统一票型，避免好人互投。",
                rationale="Coordinate the village without making an unverifiable role claim.",
            ),
        ]

    def _village_discussion_candidates(
        self, observation: AgentObservation
    ) -> list[CandidateAction]:
        targets = self._rank_discussion_targets(observation)
        target = targets[0] if targets else "当前高风险玩家"
        return [
            CandidateAction(
                ActionType.SPEAK,
                "evidence_accusation",
                message=f"我当前最怀疑 {target}，请他解释自己的发言变化和投票收益。",
                rationale="Ask for a falsifiable explanation instead of making a bare accusation.",
            ),
            CandidateAction(
                ActionType.SPEAK,
                "information_request",
                message="请每个人明确给出一名怀疑对象、理由和预期票型，避免空泛发言。",
                rationale="Increase public information before voting.",
            ),
            CandidateAction(
                ActionType.SPEAK,
                "vote_coordination",
                message=f"临近投票不要分散，我建议先围绕 {target} 对齐证据和票型。",
                rationale="Reduce coordination failure on the village side.",
            ),
        ]


class LLMPlanner:
    def __init__(self, backend: ChatBackend, fallback: HeuristicPlanner) -> None:
        self.backend = backend
        self.fallback = fallback

    async def plan(self, observation: AgentObservation) -> list[CandidateAction]:
        expected = expected_action_type(observation.phase)
        legal_targets = ", ".join(_legal_targets(observation)) or "none"
        system = (
            "You are the Planner in a Werewolf agent. Return exactly three strategically distinct "
            "candidate actions as strict JSON. Use only visible information, never target a player "
            "outside legal_targets, never expose a hidden Werewolf/Doctor role, and keep claims "
            "consistent with private checks."
        )
        prompt = (
            observation.prompt() + '\nReturn {"candidates":[{"action_type":...,"strategy":...,'
            '"target_id":null-or-id,"message":...,"rationale":...}]}. '
            f"Expected action type: {expected.value}. Legal targets: [{legal_targets}]. "
            "For speech, target_id must be null and message must be non-empty."
        )
        try:
            payload = await self.backend.generate_json(system=system, prompt=prompt)
            parsed = self._parse_payload(observation, payload, expected)
        except Exception:
            return await self.fallback.plan(observation)

        fallback_candidates = await self.fallback.plan(observation)
        combined: list[CandidateAction] = []
        seen: set[tuple[ActionType, str | None, str]] = set()
        for candidate in [*parsed, *fallback_candidates]:
            signature = _candidate_signature(candidate)
            if signature in seen or not _candidate_is_legal(observation, candidate, expected):
                continue
            combined.append(candidate)
            seen.add(signature)
            if len(combined) == 3:
                break
        return combined or fallback_candidates

    @staticmethod
    def _parse_payload(
        observation: AgentObservation,
        payload: object,
        expected: ActionType,
    ) -> list[CandidateAction]:
        if not isinstance(payload, Mapping):
            return []
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            return []

        parsed: list[CandidateAction] = []
        seen: set[tuple[ActionType, str | None, str]] = set()
        for index, item in enumerate(raw_candidates):
            candidate = LLMPlanner._parse_candidate(item, expected, index)
            if candidate is None or not _candidate_is_legal(observation, candidate, expected):
                continue
            signature = _candidate_signature(candidate)
            if signature in seen:
                continue
            parsed.append(candidate)
            seen.add(signature)
        return parsed

    @staticmethod
    def _parse_candidate(item: object, expected: ActionType, index: int) -> CandidateAction | None:
        if not isinstance(item, Mapping):
            return None
        raw_action_type = item.get("action_type", expected.value)
        if isinstance(raw_action_type, ActionType):
            action_type = raw_action_type
        elif isinstance(raw_action_type, str):
            try:
                action_type = ActionType(raw_action_type.strip().casefold())
            except ValueError:
                return None
        else:
            return None

        raw_target_id = item.get("target_id")
        if raw_target_id is None:
            target_id = None
        elif isinstance(raw_target_id, str):
            target_id = raw_target_id.strip() or None
        else:
            return None

        raw_message = item.get("message", "")
        if raw_message is None:
            message = ""
        elif isinstance(raw_message, str):
            message = raw_message.strip()
        else:
            return None

        raw_strategy = item.get("strategy", "")
        strategy = (
            raw_strategy.strip()
            if isinstance(raw_strategy, str) and raw_strategy.strip()
            else f"llm_candidate_{index + 1}"
        )
        raw_rationale = item.get("rationale", "")
        rationale = raw_rationale.strip() if isinstance(raw_rationale, str) else ""
        return CandidateAction(
            action_type=action_type,
            strategy=strategy,
            target_id=target_id,
            message=message,
            rationale=rationale,
        )


class StrategicEvaluator:
    async def evaluate(
        self, observation: AgentObservation, candidate: CandidateAction
    ) -> CandidateEvaluation:
        expected = expected_action_type(observation.phase)
        legality_reasons = _legality_reasons(observation, candidate, expected)
        if legality_reasons:
            return CandidateEvaluation(score=-1.0, legal=False, reasons=legality_reasons)

        reasons: list[str] = []
        score = 0.45
        base_strategy = candidate.strategy.removeprefix("reflexion_")

        if candidate.action_type is ActionType.ABSTAIN:
            score -= 0.35 if observation.legal_targets else 0.15
            reasons.append("Abstention preserves legality but yields little strategic value.")

        if candidate.target_id:
            target_id = candidate.target_id
            wolf_probability = _wolf_probability(observation, target_id)
            if observation.role is Role.WEREWOLF:
                score += 0.2 * (1.0 - wolf_probability)
                if target_id in observation.teammate_ids:
                    score -= 1.5
                    reasons.append("Avoid targeting a known Werewolf teammate.")
                if expected is ActionType.KILL:
                    score += 0.35 * self._seer_claim_bonus(observation, target_id)
                    score += 0.05 * _speaker_influence(observation, target_id)
                if expected is ActionType.VOTE:
                    declared_target = _latest_declared_target(
                        observation, sender=observation.player_id
                    )
                    score += 0.25 * (target_id == declared_target)
            elif expected is ActionType.VOTE:
                score += 0.5 * wolf_probability
                score += 0.04 * _public_pressure(observation, target_id)

            if expected is ActionType.CHECK:
                has_belief = _has_role_belief(observation, target_id, Role.WEREWOLF)
                confirmed = has_belief and (wolf_probability <= 0.01 or wolf_probability >= 0.99)
                if confirmed:
                    score -= 0.45
                    reasons.append("Avoid repeating a check whose alignment is already confirmed.")
                else:
                    uncertainty = (
                        1.0 if not has_belief else max(0.0, 1.0 - abs(wolf_probability - 0.5) * 2.0)
                    )
                    score += 0.3 * uncertainty + 0.2 * wolf_probability
                if target_id in _seer_claimants(observation):
                    score += 0.25
                    reasons.append("Checking a competing Seer claim resolves a key contradiction.")

            if expected is ActionType.PROTECT:
                score += self._seer_claim_bonus(observation, target_id)
                score += 0.1 * (target_id == observation.player_id)
                score -= 0.35 * wolf_probability
                if target_id == _last_protected_target(observation):
                    score -= 0.08
                    reasons.append("Repeated protection is predictable when alternatives exist.")

        score += self._strategy_bonus(observation, base_strategy)

        if candidate.action_type is ActionType.SPEAK:
            message = candidate.message
            if observation.role is Role.WEREWOLF:
                if "我是狼人" in message:
                    score -= 2.0
                    reasons.append("The message leaks the hidden Werewolf role.")
                if base_strategy in {"fake_seer_claim", "lead_vote"} and any(
                    _message_mentions(message, teammate_id)
                    for teammate_id in observation.teammate_ids
                ):
                    score -= 1.5
                    reasons.append("The public push exposes or sacrifices a known teammate.")
            if observation.role is Role.DOCTOR and (
                "我是医生" in message or "i am the doctor" in message.casefold()
            ):
                score -= 1.2
                reasons.append("The Doctor should avoid unnecessary public role exposure.")
            if observation.role is Role.SEER:
                for known_villager in _known_villagers(observation):
                    if _message_mentions(message, known_villager) and "狼人" in message:
                        score -= 1.2
                        reasons.append("The statement contradicts a confirmed non-wolf check.")

        score += self._stable_jitter(observation, candidate)
        reasons.append(f"Strategic heuristic score={score:.3f}.")
        return CandidateEvaluation(score=score, legal=True, reasons=tuple(reasons))

    @staticmethod
    def _strategy_bonus(observation: AgentObservation, strategy: str) -> float:
        common_bonuses = {
            "coordinate_vote": 0.28,
            "vote_coordination": 0.16,
            "reveal_and_accuse": 0.3,
            "counter_fake_seer": 0.24,
            "evidence_chain": 0.18,
            "information_request": 0.08,
            "share_golden_water": 0.16,
            "protect_information_flow": 0.1,
            "evidence_vote": 0.12,
            "vote_known_wolf": 0.35,
            "check_counterclaim": 0.22,
            "check_suspect": 0.14,
            "check_unknown": 0.12,
            "check_information_gain": 0.1,
            "protect_seer_claim": 0.3,
            "protect_influencer": 0.1,
            "self_preservation": 0.06,
            "kill_seer_claimant": 0.28,
            "kill_influencer": 0.16,
            "abstain": -0.1,
        }
        bonus = common_bonuses.get(strategy, 0.0)
        if observation.role is Role.WEREWOLF:
            wolf_bonuses = {
                "fake_seer_claim": 0.3 if observation.round_no >= 2 else 0.22,
                "lead_vote": 0.24,
                "follow_through_vote": 0.32,
                "build_wagon": 0.16,
                "conceal_role": 0.1,
            }
            bonus += wolf_bonuses.get(strategy, 0.0)
        if observation.role is Role.DOCTOR and strategy == "conceal_doctor":
            bonus += 0.12
        return bonus

    @staticmethod
    def _is_legal(
        observation: AgentObservation, candidate: CandidateAction, expected: ActionType
    ) -> bool:
        return _candidate_is_legal(observation, candidate, expected)

    @staticmethod
    def _seer_claim_bonus(observation: AgentObservation, target_id: str) -> float:
        return 0.45 if target_id in _seer_claimants(observation) else 0.0

    @staticmethod
    def _stable_jitter(observation: AgentObservation, candidate: CandidateAction) -> float:
        value = _stable_integer(
            observation.game_id,
            observation.round_no,
            observation.player_id,
            observation.phase.value,
            candidate.action_type.value,
            candidate.strategy,
            candidate.target_id,
            candidate.message,
        )
        return (value % 1000) / 1_000_000.0


class RuleExecutor:
    def execute(self, observation: AgentObservation, candidate: CandidateAction) -> GameAction:
        legality_reasons = _legality_reasons(observation, candidate)
        if legality_reasons:
            reason_text = "; ".join(legality_reasons)
            raise ValueError(f"cannot execute illegal candidate: {reason_text}")
        target_id = None if candidate.action_type is ActionType.ABSTAIN else candidate.target_id
        return GameAction(
            actor_id=observation.player_id,
            action_type=candidate.action_type,
            target_id=target_id,
            message=candidate.message,
            strategy=candidate.strategy,
        )


class RuleReflexion:
    async def revise(
        self,
        observation: AgentObservation,
        candidate: CandidateAction,
        evaluation: CandidateEvaluation,
    ) -> tuple[CandidateAction, str]:
        seed = _stable_integer(
            "reflexion",
            observation.game_id,
            observation.round_no,
            observation.player_id,
            observation.phase.value,
        )
        alternatives = await HeuristicPlanner(seed=seed).plan(observation)
        legal_alternatives = [
            alternative
            for alternative in alternatives
            if _candidate_is_legal(observation, alternative)
        ]
        original_signature = _candidate_signature(candidate)
        replacement = next(
            (
                alternative
                for alternative in legal_alternatives
                if _candidate_signature(alternative) != original_signature
            ),
            legal_alternatives[0] if legal_alternatives else self._emergency_candidate(observation),
        )
        strategy = replacement.strategy
        if not strategy.startswith("reflexion_"):
            strategy = f"reflexion_{strategy}"
        revised = CandidateAction(
            action_type=replacement.action_type,
            strategy=strategy,
            target_id=replacement.target_id,
            message=replacement.message,
            rationale=(
                f"Reflexion repair of '{candidate.strategy}'. {replacement.rationale}"
            ).strip(),
        )
        reason_text = "; ".join(evaluation.reasons) or (
            f"evaluator score {evaluation.score:.3f} was below the reflection threshold"
        )
        reflection = (
            f"Reflexion replaced '{candidate.strategy}' with '{revised.strategy}' because: "
            f"{reason_text}"
        )
        return revised, reflection

    @staticmethod
    def _emergency_candidate(observation: AgentObservation) -> CandidateAction:
        expected = expected_action_type(observation.phase)
        targets = _legal_targets(observation)
        if expected is ActionType.SPEAK:
            return CandidateAction(
                ActionType.SPEAK,
                "reflexion_safe_speech",
                message="我只基于公开信息发言：请大家明确怀疑对象、证据和最终票型。",
                rationale="Use a deterministic non-role-leaking public statement.",
            )
        if targets:
            return CandidateAction(
                expected,
                "reflexion_legal_target",
                target_id=targets[0],
                rationale="Select the first deterministically sorted legal target.",
            )
        return CandidateAction(
            ActionType.ABSTAIN,
            "reflexion_no_legal_target",
            rationale="No legal target exists.",
        )


@dataclass(slots=True)
class CognitiveAgent:
    planner: Planner
    evaluator: Evaluator
    executor: Executor
    reflexion: Reflexion
    reflection_threshold: float = 0.15

    async def decide(self, observation: AgentObservation) -> DecisionTrace:
        candidates = list(await self.planner.plan(observation))
        if not candidates:
            raise RuntimeError("planner returned no candidates")
        evaluations = list(
            await asyncio.gather(
                *(self.evaluator.evaluate(observation, candidate) for candidate in candidates)
            )
        )
        selected_index = self._select_index(observation, candidates, evaluations)
        selected_evaluation = evaluations[selected_index]
        reflection = ""

        if (
            not self._effectively_legal(
                observation, candidates[selected_index], selected_evaluation
            )
            or selected_evaluation.score < self.reflection_threshold
        ):
            revised, reflection = await self.reflexion.revise(
                observation, candidates[selected_index], selected_evaluation
            )
            candidates.append(revised)
            evaluations.append(await self.evaluator.evaluate(observation, revised))
            selected_index = self._select_index(observation, candidates, evaluations)

        if not self._effectively_legal(
            observation, candidates[selected_index], evaluations[selected_index]
        ):
            emergency, emergency_reflection = await RuleReflexion().revise(
                observation, candidates[selected_index], evaluations[selected_index]
            )
            candidates.append(emergency)
            evaluations.append(await self.evaluator.evaluate(observation, emergency))
            selected_index = self._select_index(observation, candidates, evaluations)
            reflection = " | ".join(text for text in (reflection, emergency_reflection) if text)

        selected = candidates[selected_index]
        selected_evaluation = evaluations[selected_index]
        if not self._effectively_legal(observation, selected, selected_evaluation):
            raise RuntimeError("Reflexion failed to produce a legal executable action")

        action = self.executor.execute(observation, selected)
        return DecisionTrace(
            player_id=observation.player_id,
            role=observation.role,
            round_no=observation.round_no,
            phase=observation.phase,
            observation_prompt=observation.prompt(),
            candidates=tuple(candidates),
            evaluations=tuple(evaluations),
            selected_index=selected_index,
            action=action,
            reflection=reflection,
        )

    @staticmethod
    def _select_index(
        observation: AgentObservation,
        candidates: list[CandidateAction],
        evaluations: list[CandidateEvaluation],
    ) -> int:
        def selection_key(index: int) -> tuple[int, float, int]:
            evaluation = evaluations[index]
            effectively_legal = CognitiveAgent._effectively_legal(
                observation, candidates[index], evaluation
            )
            score = evaluation.score if math.isfinite(evaluation.score) else -math.inf
            return int(effectively_legal), score, -index

        return max(range(len(candidates)), key=selection_key)

    @staticmethod
    def _effectively_legal(
        observation: AgentObservation,
        candidate: CandidateAction,
        evaluation: CandidateEvaluation,
    ) -> bool:
        return evaluation.legal and _candidate_is_legal(observation, candidate)


def build_agent(seed: int, backend: ChatBackend | None = None) -> CognitiveAgent:
    heuristic_planner = HeuristicPlanner(seed=seed)
    planner: Planner
    if backend is None:
        planner = heuristic_planner
    else:
        planner = LLMPlanner(backend=backend, fallback=heuristic_planner)
    return CognitiveAgent(
        planner=planner,
        evaluator=StrategicEvaluator(),
        executor=RuleExecutor(),
        reflexion=RuleReflexion(),
    )
