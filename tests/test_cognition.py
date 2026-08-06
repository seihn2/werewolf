import pytest

from wolfplay.cognition import (
    CognitiveAgent,
    HeuristicPlanner,
    LLMPlanner,
    RuleExecutor,
    RuleReflexion,
    StrategicEvaluator,
)
from wolfplay.models import (
    ActionType,
    AgentObservation,
    CandidateAction,
    CandidateEvaluation,
    GameEvent,
    Phase,
    Role,
)


def make_observation(
    *,
    role: Role = Role.VILLAGER,
    phase: Phase = Phase.DAY_DISCUSSION,
    player_id: str = "player_0",
    legal_targets: tuple[str, ...] = ("player_1", "player_2", "player_3"),
    teammate_ids: tuple[str, ...] = (),
    role_beliefs: dict[str, dict[str, float]] | None = None,
    events: tuple[GameEvent, ...] = (),
    round_no: int = 2,
) -> AgentObservation:
    player_ids = tuple(dict.fromkeys((player_id, *legal_targets, *teammate_ids)))
    return AgentObservation(
        game_id="game",
        player_id=player_id,
        player_name=player_id,
        role=role,
        teammate_ids=teammate_ids,
        alive_players=tuple(
            {"player_id": current_id, "name": current_id, "alive": True}
            for current_id in player_ids
        ),
        round_no=round_no,
        phase=phase,
        legal_targets=legal_targets,
        events=events,
        memory_context="",
        role_beliefs=role_beliefs or {},
    )


def speech_event(
    sender: str,
    message: str,
    *,
    logical_time: int = 1,
    strategy: str = "",
) -> GameEvent:
    return GameEvent(
        logical_time=logical_time,
        topic="speech",
        round_no=2,
        phase=Phase.DAY_DISCUSSION,
        payload={"message": message, "strategy": strategy},
        sender=sender,
    )


class StaticPlanner:
    def __init__(self, candidates: list[CandidateAction]) -> None:
        self.candidates = candidates

    async def plan(self, observation: AgentObservation) -> list[CandidateAction]:
        return list(self.candidates)


class StaticBackend:
    def __init__(self, payload: object = None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error

    async def generate_json(self, *, system: str, prompt: str):
        if self.error is not None:
            raise self.error
        return self.payload


class LegalityFirstEvaluator:
    async def evaluate(
        self, observation: AgentObservation, candidate: CandidateAction
    ) -> CandidateEvaluation:
        if candidate.target_id == "ghost":
            return CandidateEvaluation(score=100.0, legal=False, reasons=("illegal",))
        return CandidateEvaluation(score=0.5, legal=True, reasons=("legal",))


async def test_werewolf_discussion_includes_fake_seer_and_vote_lead_without_teammate():
    observation = make_observation(
        role=Role.WEREWOLF,
        teammate_ids=("player_1",),
        role_beliefs={
            "player_2": {Role.WEREWOLF.value: 0.2},
            "player_3": {Role.WEREWOLF.value: 0.1},
        },
    )

    candidates = await HeuristicPlanner(seed=7).plan(observation)

    strategies = {candidate.strategy for candidate in candidates}
    assert {"fake_seer_claim", "lead_vote", "conceal_role"} <= strategies
    fake_claim = next(
        candidate for candidate in candidates if candidate.strategy == "fake_seer_claim"
    )
    vote_lead = next(candidate for candidate in candidates if candidate.strategy == "lead_vote")
    assert "预言家" in fake_claim.message
    assert "票归" in fake_claim.message
    assert "带票" in vote_lead.rationale
    assert all("player_1" not in candidate.message for candidate in candidates)


async def test_heuristic_planner_is_deterministic_for_same_observation():
    observation = make_observation(
        role=Role.SEER,
        phase=Phase.NIGHT_SEER,
        role_beliefs={"player_1": {Role.WEREWOLF.value: 0.4}},
    )
    planner = HeuristicPlanner(seed=17)

    first = await planner.plan(observation)
    second = await planner.plan(observation)
    same_seed = await HeuristicPlanner(seed=17).plan(observation)

    assert first == second == same_seed


async def test_werewolf_vote_follows_through_on_public_vote_lead():
    observation = make_observation(
        role=Role.WEREWOLF,
        phase=Phase.DAY_VOTE,
        teammate_ids=("player_1",),
        events=(
            speech_event(
                "player_0",
                "我今天明确带票 player_3，所有人统一投 player_3。",
                strategy="lead_vote",
            ),
        ),
    )

    candidates = await HeuristicPlanner(seed=23).plan(observation)

    assert candidates[0].target_id == "player_3"
    assert candidates[0].strategy == "follow_through_vote"
    assert all(candidate.target_id != "player_1" for candidate in candidates)


async def test_seer_prefers_unconfirmed_suspect_over_rechecking_confirmed_players():
    observation = make_observation(
        role=Role.SEER,
        phase=Phase.NIGHT_SEER,
        role_beliefs={
            "player_1": {Role.WEREWOLF.value: 1.0},
            "player_2": {Role.WEREWOLF.value: 0.0},
            "player_3": {Role.WEREWOLF.value: 0.65},
        },
    )

    candidates = await HeuristicPlanner(seed=3).plan(observation)

    assert candidates[0].target_id == "player_3"
    assert candidates[0].strategy == "check_suspect"


async def test_seer_discussion_reports_confirmed_wolf_without_inventing_target():
    observation = make_observation(
        role=Role.SEER,
        role_beliefs={
            "player_1": {Role.WEREWOLF.value: 0.0},
            "player_2": {Role.WEREWOLF.value: 1.0},
        },
    )

    candidates = await HeuristicPlanner(seed=31).plan(observation)

    reveal = next(
        candidate for candidate in candidates if candidate.strategy == "reveal_and_accuse"
    )
    assert "player_2" in reveal.message
    assert "player_1 是狼人" not in reveal.message


async def test_doctor_prioritizes_public_seer_claimant():
    observation = make_observation(
        role=Role.DOCTOR,
        phase=Phase.NIGHT_DOCTOR,
        legal_targets=("player_0", "player_1", "player_2"),
        events=(speech_event("player_2", "我是预言家，昨晚查验 player_1 是狼人。"),),
    )

    candidates = await HeuristicPlanner(seed=11).plan(observation)

    assert candidates[0].target_id == "player_2"
    assert candidates[0].strategy == "protect_seer_claim"


async def test_heuristic_candidates_are_legal_across_roles_and_decision_phases():
    decision_phases = (
        Phase.NIGHT_WEREWOLF,
        Phase.NIGHT_SEER,
        Phase.NIGHT_DOCTOR,
        Phase.DAY_DISCUSSION,
        Phase.DAY_VOTE,
    )
    evaluator = StrategicEvaluator()

    for role in Role:
        for phase in decision_phases:
            observation = make_observation(
                role=role,
                phase=phase,
                teammate_ids=("player_1",) if role is Role.WEREWOLF else (),
            )
            candidates = await HeuristicPlanner(seed=41).plan(observation)
            evaluations = [
                await evaluator.evaluate(observation, candidate) for candidate in candidates
            ]

            assert candidates
            assert all(evaluation.legal for evaluation in evaluations)


async def test_llm_planner_falls_back_on_malformed_payload_and_backend_error():
    observation = make_observation(role=Role.SEER, phase=Phase.NIGHT_SEER)
    expected = await HeuristicPlanner(seed=5).plan(observation)

    malformed = LLMPlanner(
        backend=StaticBackend(payload={"candidates": "not-a-list"}),
        fallback=HeuristicPlanner(seed=5),
    )
    failed = LLMPlanner(
        backend=StaticBackend(error=RuntimeError("backend unavailable")),
        fallback=HeuristicPlanner(seed=5),
    )

    assert await malformed.plan(observation) == expected
    assert await failed.plan(observation) == expected


async def test_llm_planner_filters_illegal_candidates_and_supplements_fallback():
    observation = make_observation(role=Role.SEER, phase=Phase.NIGHT_SEER)
    planner = LLMPlanner(
        backend=StaticBackend(
            payload={
                "candidates": [
                    {
                        "action_type": "kill",
                        "strategy": "wrong_phase",
                        "target_id": "player_1",
                    },
                    {
                        "action_type": "check",
                        "strategy": "llm_valid",
                        "target_id": "player_2",
                    },
                    {
                        "action_type": "check",
                        "strategy": "illegal_target",
                        "target_id": "ghost",
                    },
                    42,
                ]
            }
        ),
        fallback=HeuristicPlanner(seed=9),
    )
    evaluator = StrategicEvaluator()

    candidates = await planner.plan(observation)
    evaluations = [await evaluator.evaluate(observation, candidate) for candidate in candidates]

    assert len(candidates) == 3
    assert candidates[0].strategy == "llm_valid"
    assert all(evaluation.legal for evaluation in evaluations)
    assert all(candidate.action_type is ActionType.CHECK for candidate in candidates)


async def test_reflexion_repairs_wrong_action_type_with_strategic_seer_target():
    observation = make_observation(
        role=Role.SEER,
        phase=Phase.NIGHT_SEER,
        legal_targets=("player_1", "player_2"),
        role_beliefs={
            "player_1": {Role.WEREWOLF.value: 0.0},
            "player_2": {Role.WEREWOLF.value: 0.7},
        },
    )
    agent = CognitiveAgent(
        planner=StaticPlanner(
            [
                CandidateAction(
                    action_type=ActionType.KILL,
                    strategy="illegal_self_target",
                    target_id=observation.player_id,
                )
            ]
        ),
        evaluator=StrategicEvaluator(),
        executor=RuleExecutor(),
        reflexion=RuleReflexion(),
    )

    trace = await agent.decide(observation)

    assert trace.reflection
    assert trace.action.action_type is ActionType.CHECK
    assert trace.action.target_id == "player_2"
    assert trace.action.strategy.startswith("reflexion_")


async def test_reflexion_moves_werewolf_vote_off_teammate():
    observation = make_observation(
        role=Role.WEREWOLF,
        phase=Phase.DAY_VOTE,
        legal_targets=("player_1", "player_2"),
        teammate_ids=("player_1",),
    )
    agent = CognitiveAgent(
        planner=StaticPlanner(
            [CandidateAction(ActionType.VOTE, "sacrifice_teammate", target_id="player_1")]
        ),
        evaluator=StrategicEvaluator(),
        executor=RuleExecutor(),
        reflexion=RuleReflexion(),
    )

    trace = await agent.decide(observation)

    assert trace.reflection
    assert trace.action.target_id == "player_2"
    assert trace.action.strategy.startswith("reflexion_")


async def test_cognitive_agent_prioritizes_legality_over_misleading_score():
    observation = make_observation(phase=Phase.DAY_VOTE, legal_targets=("player_1",))
    agent = CognitiveAgent(
        planner=StaticPlanner(
            [
                CandidateAction(ActionType.VOTE, "illegal_high_score", target_id="ghost"),
                CandidateAction(ActionType.VOTE, "legal_lower_score", target_id="player_1"),
            ]
        ),
        evaluator=LegalityFirstEvaluator(),
        executor=RuleExecutor(),
        reflexion=RuleReflexion(),
        reflection_threshold=0.0,
    )

    trace = await agent.decide(observation)

    assert trace.action.target_id == "player_1"
    assert trace.selected_index == 1
    assert not trace.reflection


async def test_empty_doctor_speech_is_repaired_without_revealing_role():
    observation = make_observation(role=Role.DOCTOR)
    agent = CognitiveAgent(
        planner=StaticPlanner([CandidateAction(ActionType.SPEAK, "empty", message="  ")]),
        evaluator=StrategicEvaluator(),
        executor=RuleExecutor(),
        reflexion=RuleReflexion(),
    )

    trace = await agent.decide(observation)

    assert trace.reflection
    assert trace.action.action_type is ActionType.SPEAK
    assert trace.action.message.strip()
    assert "我是医生" not in trace.action.message
    assert trace.action.strategy == "reflexion_conceal_doctor"


async def test_no_legal_target_uses_deterministic_abstain():
    observation = make_observation(
        role=Role.SEER,
        phase=Phase.NIGHT_SEER,
        legal_targets=(),
    )
    planner = HeuristicPlanner(seed=1)
    evaluator = StrategicEvaluator()

    candidate = (await planner.plan(observation))[0]
    evaluation = await evaluator.evaluate(observation, candidate)
    action = RuleExecutor().execute(observation, candidate)

    assert candidate.action_type is ActionType.ABSTAIN
    assert evaluation.legal
    assert action.target_id is None


def test_executor_rejects_illegal_candidate():
    observation = make_observation(phase=Phase.DAY_VOTE, legal_targets=("player_1",))
    candidate = CandidateAction(ActionType.VOTE, "bad", target_id="ghost")

    with pytest.raises(ValueError, match="cannot execute illegal candidate"):
        RuleExecutor().execute(observation, candidate)
