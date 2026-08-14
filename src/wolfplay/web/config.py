from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class WebSettings:
    data_dir: Path
    artifact_dir: Path
    database_url: str
    frontend_dist: Path
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    max_concurrent_games: int = 4
    max_concurrent_jobs: int = 1
    realtime_queue_size: int = 512
    heartbeat_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> WebSettings:
        root = _project_root()
        data_dir = Path(
            os.getenv("WOLFPLAY_STUDIO_DATA_DIR", str(root / ".wolfplay-studio"))
        ).expanduser()
        artifact_dir = Path(
            os.getenv("WOLFPLAY_STUDIO_ARTIFACT_DIR", str(data_dir / "artifacts"))
        ).expanduser()
        database_url = os.getenv(
            "WOLFPLAY_STUDIO_DATABASE_URL",
            f"sqlite+aiosqlite:///{(data_dir / 'wolfplay.db').resolve()}",
        )
        frontend_dist = Path(
            os.getenv("WOLFPLAY_STUDIO_FRONTEND_DIST", str(root / "web" / "dist"))
        ).expanduser()
        origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "WOLFPLAY_STUDIO_CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        )
        heartbeat_seconds = float(os.getenv("WOLFPLAY_STUDIO_HEARTBEAT_SECONDS", "15"))
        if heartbeat_seconds <= 0:
            raise ValueError("WOLFPLAY_STUDIO_HEARTBEAT_SECONDS must be positive")
        return cls(
            data_dir=data_dir,
            artifact_dir=artifact_dir,
            database_url=database_url,
            frontend_dist=frontend_dist,
            host=os.getenv("WOLFPLAY_STUDIO_HOST", "127.0.0.1"),
            port=_positive_int("WOLFPLAY_STUDIO_PORT", 8000),
            log_level=os.getenv("WOLFPLAY_STUDIO_LOG_LEVEL", "info"),
            cors_origins=origins,
            max_concurrent_games=_positive_int("WOLFPLAY_STUDIO_MAX_GAMES", 4),
            max_concurrent_jobs=_positive_int("WOLFPLAY_STUDIO_MAX_JOBS", 1),
            realtime_queue_size=_positive_int("WOLFPLAY_STUDIO_QUEUE_SIZE", 512),
            heartbeat_seconds=heartbeat_seconds,
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def resolve_artifact_path(self, relative_path: str | Path) -> Path:
        root = self.artifact_dir.resolve()
        candidate = (root / Path(relative_path)).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("artifact path must stay inside the configured artifact directory")
        return candidate
