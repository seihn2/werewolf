import time
from pathlib import Path

from fastapi.testclient import TestClient

from wolfplay.web.app import create_app
from wolfplay.web.config import WebSettings


def build_settings(tmp_path: Path) -> WebSettings:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return WebSettings(
        data_dir=tmp_path,
        artifact_dir=artifact_dir,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'api.db'}",
        frontend_dist=tmp_path / "missing-dist",
        heartbeat_seconds=0.1,
    )


def wait_for_game(client: TestClient, game_id: str) -> dict:
    for _ in range(200):
        game = client.get(f"/api/games/{game_id}").json()
        if game["status"] in {"completed", "failed", "cancelled"}:
            return game
        time.sleep(0.01)
    raise AssertionError("game did not complete")


def wait_for_job(client: TestClient, job_id: str) -> dict:
    for _ in range(300):
        job = client.get(f"/api/training/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not complete")


def test_game_api_enforces_live_private_view_and_persists_replay(tmp_path):
    with TestClient(create_app(build_settings(tmp_path))) as client:
        response = client.post(
            "/api/games",
            json={"seed": 17, "max_rounds": 2, "pace_seconds": 0.02},
        )
        assert response.status_code == 202
        game_id = response.json()["id"]

        private_live = client.get(f"/api/games/{game_id}/events?view=omniscient")
        assert private_live.status_code == 409

        with client.websocket_connect(f"/ws/games/{game_id}") as socket:
            snapshot = socket.receive_json()
            assert snapshot["type"] == "snapshot"
            assert snapshot["game"]["id"] == game_id

        completed = wait_for_game(client, game_id)
        public_events = client.get(f"/api/games/{game_id}/events").json()
        all_events = client.get(f"/api/games/{game_id}/events?view=omniscient").json()

        assert completed["status"] == "completed"
        assert completed["created_at"].endswith("Z")
        assert completed["result"]["decision_traces"]
        assert all(event["audience"] is None for event in public_events)
        assert any(event["topic"] == "role_assignment" for event in all_events)
        assert len(all_events) > len(public_events)


def test_agent_and_analytics_api(tmp_path):
    with TestClient(create_app(build_settings(tmp_path))) as client:
        created = client.post(
            "/api/agents",
            json={
                "name": "Local strategist",
                "kind": "openai_compatible",
                "base_url": "http://127.0.0.1:9000/v1",
                "model": "local-model",
                "env_prefix": "WOLFPLAY_LOCAL",
            },
        )
        assert created.status_code == 201
        agent = created.json()
        updated = client.patch(
            f"/api/agents/{agent['id']}",
            json={"temperature": 0.25},
        )
        assert updated.status_code == 200
        assert updated.json()["temperature"] == 0.25
        assert client.get("/api/analytics/overview").status_code == 200
        assert client.get("/api/health").json()["status"] == "ok"


def test_self_play_training_job_runs_through_api(tmp_path):
    with TestClient(create_app(build_settings(tmp_path))) as client:
        response = client.post(
            "/api/training/jobs",
            json={
                "kind": "self_play",
                "config": {
                    "games": 1,
                    "concurrency": 1,
                    "seed": 55,
                    "max_rounds": 1,
                    "backend": "heuristic",
                },
            },
        )
        assert response.status_code == 202, response.text
        job = wait_for_job(client, response.json()["id"])
        logs = client.get(f"/api/training/jobs/{job['id']}/logs").json()

        assert job["status"] == "completed", job["error"]
        assert job["metrics"]["games"] == 1
        assert Path(job["output_path"]).is_file()
        assert logs["lines"]
        artifacts = client.get("/api/artifacts").json()["items"]
        trajectory = next(item for item in artifacts if item["category"] == "self_play")
        download = client.get(f"/api/artifacts/{trajectory['path']}/download")
        assert download.status_code == 200
