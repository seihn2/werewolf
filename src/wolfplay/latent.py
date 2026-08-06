from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from .models import Role

_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


class TextEmbedder(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def name(self) -> str: ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class HashingTextEmbedder:
    """Deterministic dependency-free text embeddings for offline runs and tests."""

    dimensions: int = 256

    def __post_init__(self) -> None:
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")

    @property
    def dimension(self) -> int:
        return self.dimensions

    @property
    def name(self) -> str:
        return f"hashing-blake2b-{self.dimensions}"

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("embedding input must be a non-empty string")
        vector = [0.0] * self.dimensions
        normalized = " ".join(text.casefold().split())
        features = list(_TOKEN_PATTERN.findall(normalized))
        compact = normalized.replace(" ", "")
        features.extend(compact[index : index + 2] for index in range(max(0, len(compact) - 1)))
        features.extend(compact[index : index + 3] for index in range(max(0, len(compact) - 2)))
        if not features:
            features = [normalized]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return _l2_normalize(vector)


@dataclass(frozen=True, slots=True)
class EmbeddingModelConfig:
    base_url: str
    api_key: str
    model: str
    dimensions: int | None = None
    timeout_seconds: float = 90.0
    batch_size: int = 128

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @classmethod
    def from_env(cls, prefix: str = "WOLFPLAY_EMBEDDING") -> EmbeddingModelConfig:
        prefix = prefix.rstrip("_")
        names = {
            "base_url": f"{prefix}_BASE_URL",
            "api_key": f"{prefix}_API_KEY",
            "model": f"{prefix}_MODEL",
        }
        missing = [name for name in names.values() if not os.getenv(name)]
        if missing:
            raise RuntimeError(f"missing environment variables: {', '.join(missing)}")
        raw_dimensions = os.getenv(f"{prefix}_DIMENSIONS")
        return cls(
            base_url=os.environ[names["base_url"]],
            api_key=os.environ[names["api_key"]],
            model=os.environ[names["model"]],
            dimensions=int(raw_dimensions) if raw_dimensions else None,
        )


class OpenAICompatibleEmbedder:
    """Synchronous client for OpenAI-compatible `/embeddings` endpoints."""

    def __init__(self, config: EmbeddingModelConfig) -> None:
        self.config = config
        self._dimension = config.dimensions

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("embedding dimension is unknown until the first request")
        return self._dimension

    @property
    def name(self) -> str:
        return self.config.model

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("embedding inputs must be non-empty strings")
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            for start in range(0, len(texts), self.config.batch_size):
                batch = list(texts[start : start + self.config.batch_size])
                payload: dict[str, Any] = {"model": self.config.model, "input": batch}
                if self.config.dimensions is not None:
                    payload["dimensions"] = self.config.dimensions
                response = client.post(
                    f"{self.config.base_url.rstrip('/')}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
                data = body.get("data")
                if not isinstance(data, list) or len(data) != len(batch):
                    raise ValueError("embedding response data length does not match the request")
                if any(not isinstance(item, dict) for item in data):
                    raise ValueError("embedding response entries must be objects")
                ordered = sorted(data, key=lambda item: item.get("index", -1))
                for item in ordered:
                    raw_vector = item.get("embedding") if isinstance(item, dict) else None
                    vector = _validated_vector(raw_vector, context="embedding response")
                    if self._dimension is None:
                        self._dimension = len(vector)
                    if len(vector) != self._dimension:
                        raise ValueError("embedding response dimensions are inconsistent")
                    vectors.append(_l2_normalize(vector))
        return vectors


@dataclass(frozen=True, slots=True)
class KMeansResult:
    labels: tuple[int, ...]
    centroids: tuple[tuple[float, ...], ...]
    inertia: float
    iterations: int


def fit_kmeans(
    vectors: Sequence[Sequence[float]],
    *,
    clusters: int,
    seed: int = 42,
    max_iterations: int = 100,
    tolerance: float = 1e-5,
) -> KMeansResult:
    points = [_validated_vector(vector, context="k-means vector") for vector in vectors]
    if not points:
        raise ValueError("k-means requires at least one vector")
    if clusters <= 0:
        raise ValueError("clusters must be positive")
    if clusters > len(points):
        raise ValueError("clusters cannot exceed the number of vectors")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative number")
    dimension = len(points[0])
    if any(len(point) != dimension for point in points):
        raise ValueError("all k-means vectors must have the same dimension")

    rng = random.Random(seed)
    centroids = _kmeans_plus_plus(points, clusters=clusters, rng=rng)
    labels = [0] * len(points)
    iterations = 0
    for iteration_number in range(1, max_iterations + 1):
        iterations = iteration_number
        labels = [_nearest_centroid(point, centroids) for point in points]
        updated: list[list[float]] = []
        for cluster_id in range(clusters):
            members = [
                point for point, label in zip(points, labels, strict=True) if label == cluster_id
            ]
            if members:
                updated.append(
                    [
                        sum(point[index] for point in members) / len(members)
                        for index in range(dimension)
                    ]
                )
            else:
                farthest_index = max(
                    range(len(points)),
                    key=lambda index: _squared_distance(points[index], centroids[labels[index]]),
                )
                updated.append(list(points[farthest_index]))
        shift = max(
            _squared_distance(old, new) for old, new in zip(centroids, updated, strict=True)
        )
        centroids = updated
        if shift <= tolerance * tolerance:
            break

    labels = [_nearest_centroid(point, centroids) for point in points]
    inertia = sum(
        _squared_distance(point, centroids[label])
        for point, label in zip(points, labels, strict=True)
    )
    return KMeansResult(
        labels=tuple(labels),
        centroids=tuple(tuple(centroid) for centroid in centroids),
        inertia=inertia,
        iterations=iterations,
    )


@dataclass(frozen=True, slots=True)
class LatentCluster:
    cluster_id: int
    centroid: tuple[float, ...]
    representative: str
    size: int
    examples: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "centroid": list(self.centroid),
            "representative": self.representative,
            "size": self.size,
            "examples": list(self.examples),
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> LatentCluster:
        return cls(
            cluster_id=int(record["cluster_id"]),
            centroid=tuple(_validated_vector(record["centroid"], context="cluster centroid")),
            representative=str(record["representative"]),
            size=int(record["size"]),
            examples=tuple(str(example) for example in record.get("examples", [])),
        )


@dataclass(frozen=True, slots=True)
class RoleLatentSpace:
    role: Role
    clusters: tuple[LatentCluster, ...]
    inertia: float
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "inertia": self.inertia,
            "samples": self.samples,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> RoleLatentSpace:
        return cls(
            role=Role(record["role"]),
            inertia=float(record["inertia"]),
            samples=int(record["samples"]),
            clusters=tuple(LatentCluster.from_dict(item) for item in record["clusters"]),
        )


@dataclass(frozen=True, slots=True)
class LatentStrategySpace:
    embedding_model: str
    embedding_dimension: int
    seed: int
    roles: dict[Role, RoleLatentSpace]
    version: int = 1

    def assign(self, role: Role, text: str, embedder: TextEmbedder) -> int:
        role_space = self.roles.get(role)
        if role_space is None or not role_space.clusters:
            raise ValueError(f"no latent strategy clusters for role {role.value}")
        vector = embedder.embed_many([text])[0]
        if len(vector) != self.embedding_dimension:
            raise ValueError(
                "embedding dimension mismatch: "
                f"expected {self.embedding_dimension}, got {len(vector)}"
            )
        return min(
            role_space.clusters,
            key=lambda cluster: (_squared_distance(vector, cluster.centroid), cluster.cluster_id),
        ).cluster_id

    def representative(self, role: Role, cluster_id: int) -> str:
        role_space = self.roles[role]
        for cluster in role_space.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster.representative
        raise KeyError(f"unknown cluster {cluster_id} for role {role.value}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "seed": self.seed,
            "roles": {
                role.value: role_space.to_dict()
                for role, role_space in sorted(self.roles.items(), key=lambda item: item[0].value)
            },
        }

    def save(self, path: Path) -> None:
        _atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> LatentStrategySpace:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("latent strategy file must contain a JSON object")
        return cls.from_dict(record)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> LatentStrategySpace:
        roles = {
            Role(role_name): RoleLatentSpace.from_dict(role_record)
            for role_name, role_record in record["roles"].items()
        }
        return cls(
            version=int(record.get("version", 1)),
            embedding_model=str(record["embedding_model"]),
            embedding_dimension=int(record["embedding_dimension"]),
            seed=int(record["seed"]),
            roles=roles,
        )


def fit_latent_strategy_space(
    records: Iterable[dict[str, Any]],
    *,
    embedder: TextEmbedder,
    clusters_by_role: dict[Role, int] | None = None,
    seed: int = 42,
    max_iterations: int = 100,
) -> LatentStrategySpace:
    configured = clusters_by_role or {
        Role.WEREWOLF: 3,
        Role.SEER: 2,
        Role.DOCTOR: 2,
        Role.VILLAGER: 2,
    }
    messages = _discussion_candidates(records)
    role_spaces: dict[Role, RoleLatentSpace] = {}
    observed_dimension: int | None = None
    for role in Role:
        role_messages = messages.get(role, [])
        if not role_messages:
            role_messages = [f"{role.value} default discussion strategy"]
        vectors = embedder.embed_many(role_messages)
        if not vectors:
            raise ValueError(f"embedder returned no vectors for role {role.value}")
        current_dimension = len(vectors[0])
        if observed_dimension is None:
            observed_dimension = current_dimension
        elif current_dimension != observed_dimension:
            raise ValueError("embedding dimensions changed between roles")
        unique_vector_count = len({tuple(vector) for vector in vectors})
        cluster_count = min(configured.get(role, 1), len(vectors), unique_vector_count)
        if cluster_count <= 0:
            raise ValueError(f"cluster count for role {role.value} must be positive")
        result = fit_kmeans(
            vectors,
            clusters=cluster_count,
            seed=seed + list(Role).index(role) * 1009,
            max_iterations=max_iterations,
        )
        clusters: list[LatentCluster] = []
        for cluster_id, centroid in enumerate(result.centroids):
            member_indices = [
                index for index, label in enumerate(result.labels) if label == cluster_id
            ]
            representative_index = min(
                member_indices,
                key=lambda index: (_squared_distance(vectors[index], centroid), index),
            )
            examples = tuple(role_messages[index] for index in member_indices[:3])
            clusters.append(
                LatentCluster(
                    cluster_id=cluster_id,
                    centroid=centroid,
                    representative=role_messages[representative_index],
                    size=len(member_indices),
                    examples=examples,
                )
            )
        role_spaces[role] = RoleLatentSpace(
            role=role,
            clusters=tuple(clusters),
            inertia=result.inertia,
            samples=len(role_messages),
        )
    if observed_dimension is None:
        raise ValueError("no embedding vectors were produced")
    return LatentStrategySpace(
        embedding_model=embedder.name,
        embedding_dimension=observed_dimension,
        seed=seed,
        roles=role_spaces,
    )


def build_latent_strategy_file(
    *,
    input_path: Path,
    output_path: Path,
    embedder: TextEmbedder,
    clusters_by_role: dict[Role, int] | None = None,
    seed: int = 42,
) -> LatentStrategySpace:
    records = load_jsonl_records(input_path)
    latent_space = fit_latent_strategy_space(
        records,
        embedder=embedder,
        clusters_by_role=clusters_by_role,
        seed=seed,
    )
    latent_space.save(output_path)
    return latent_space


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"JSONL file does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}: line {line_number}, column {error.colno}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {line_number} must be a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"JSONL file contains no records: {path}")
    return records


def _discussion_candidates(records: Iterable[dict[str, Any]]) -> dict[Role, list[str]]:
    messages = {role: [] for role in Role}
    for record_index, record in enumerate(records, start=1):
        traces = record.get("decision_traces")
        if not isinstance(traces, list):
            raise ValueError(f"record {record_index}.decision_traces must be a list")
        for trace_index, trace in enumerate(traces, start=1):
            if not isinstance(trace, dict):
                raise ValueError(f"record {record_index}.trace {trace_index} must be an object")
            role = Role(trace["role"])
            candidates = trace.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError(
                    f"record {record_index}.trace {trace_index}.candidates must be a list"
                )
            for candidate in candidates:
                if not isinstance(candidate, dict) or candidate.get("action_type") != "speak":
                    continue
                message = candidate.get("message")
                if isinstance(message, str) and message.strip():
                    messages[role].append(message.strip())
    return messages


def _kmeans_plus_plus(
    points: Sequence[Sequence[float]], *, clusters: int, rng: random.Random
) -> list[list[float]]:
    chosen = [rng.randrange(len(points))]
    while len(chosen) < clusters:
        distances = [
            min(_squared_distance(point, points[index]) for index in chosen) for point in points
        ]
        total = sum(distances)
        if total <= 0:
            next_index = next(index for index in range(len(points)) if index not in chosen)
        else:
            threshold = rng.random() * total
            cumulative = 0.0
            next_index = len(points) - 1
            for index, distance in enumerate(distances):
                cumulative += distance
                if cumulative >= threshold:
                    next_index = index
                    break
            if next_index in chosen:
                next_index = max(
                    (index for index in range(len(points)) if index not in chosen),
                    key=lambda index: distances[index],
                )
        chosen.append(next_index)
    return [list(points[index]) for index in chosen]


def _nearest_centroid(point: Sequence[float], centroids: Sequence[Sequence[float]]) -> int:
    return min(
        range(len(centroids)),
        key=lambda index: (_squared_distance(point, centroids[index]), index),
    )


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions do not match")
    return sum((first - second) ** 2 for first, second in zip(left, right, strict=True))


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _validated_vector(value: Any, *, context: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{context} must be a non-empty numeric vector")
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{context} must contain only finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{context} must contain only finite numbers")
        vector.append(number)
    return vector


def _atomic_write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
