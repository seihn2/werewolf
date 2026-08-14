# WolfPlay Studio Implementation Plan

> **For Codex:** Implement this plan task-by-task and keep the existing CLI and training behavior backward compatible.

**Goal:** Build a persistent, real-time, multi-page product around WolfPlay's game and training engines.

**Architecture:** Keep the LangGraph engine as a domain layer and attach an optional event observer. Add an asynchronous FastAPI/SQLAlchemy platform for persistence and orchestration, then build a React/Vite product UI consuming REST and WebSocket APIs.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2, aiosqlite, React 19, TypeScript, Vite, React Router, TanStack Query, Recharts, Vitest.

---

### Task 1: Runtime event bridge

**Files:**
- Modify: `src/wolfplay/engine.py`
- Test: `tests/test_engine.py`

1. Add a typed optional asynchronous event observer and optional public event pacing configuration.
2. Notify the observer after the message bus and memory store accept each event.
3. Ensure observer failures propagate to the game manager while default CLI behavior remains unchanged.
4. Add tests for ordered observation and one-shot runtime behavior.

### Task 2: Persistent application core

**Files:**
- Create: `src/wolfplay/web/config.py`
- Create: `src/wolfplay/web/database.py`
- Create: `src/wolfplay/web/tables.py`
- Create: `src/wolfplay/web/repository.py`
- Test: `tests/web/test_repository.py`

1. Add application settings and safe data/artifact path resolution.
2. Define game, event, agent profile and training job tables.
3. Initialize SQLite with WAL and foreign keys.
4. Implement repository methods for lifecycle updates, pagination and analytics queries.
5. Test persistence across independent sessions and interruption recovery.

### Task 3: Realtime game orchestration

**Files:**
- Create: `src/wolfplay/web/realtime.py`
- Create: `src/wolfplay/web/game_manager.py`
- Test: `tests/web/test_game_manager.py`

1. Implement channel-based WebSocket subscriptions with snapshots and bounded queues.
2. Start games as managed asyncio tasks and persist every event.
3. Broadcast public events only while a game is active.
4. Persist complete results and close model backends on success, failure or cancellation.
5. Test concurrent isolation, cancellation and private-event filtering.

### Task 4: Agent and training orchestration

**Files:**
- Create: `src/wolfplay/web/training.py`
- Create: `src/wolfplay/web/commands.py`
- Test: `tests/web/test_training.py`

1. Add built-in heuristic and environment-referenced OpenAI-compatible profiles.
2. Build training commands from validated typed configurations, never raw shell input.
3. Run child processes with streamed stdout/stderr, cancellation and persisted logs.
4. Support self-play, latent clustering, Deep CFR, CFR-DPO, DPO and iterative jobs.
5. Test command construction, success, failure and cancellation with lightweight commands.

### Task 5: FastAPI product API

**Files:**
- Create: `src/wolfplay/web/schemas.py`
- Create: `src/wolfplay/web/routes/*.py`
- Create: `src/wolfplay/web/app.py`
- Modify: `pyproject.toml`
- Test: `tests/web/test_api.py`

1. Add lifespan startup, database initialization, managers and interruption recovery.
2. Add health, game, event, agent, training and analytics REST routes.
3. Add game and training WebSocket routes with initial snapshots and heartbeats.
4. Add consistent error envelopes and request validation.
5. Mount production frontend assets with SPA fallback when `web/dist` exists.

### Task 6: Product frontend foundation

**Files:**
- Create: `web/package.json`
- Create: `web/vite.config.ts`
- Create: `web/src/main.tsx`
- Create: `web/src/app/*`
- Create: `web/src/styles/*`

1. Configure TypeScript, Vite, routing, API client and query cache.
2. Implement the midnight theatre design system, responsive shell and accessibility states.
3. Add shared loading, empty, error, connection and confirmation components.
4. Add typed REST models and reconnecting WebSocket hooks.

### Task 7: Arena and replay experience

**Files:**
- Create: `web/src/pages/ArenaPage.tsx`
- Create: `web/src/pages/ReplaysPage.tsx`
- Create: `web/src/features/game/*`

1. Implement game creation with faction Agent assignment and pacing.
2. Render the seven-seat table, stage state, event feed and vote relationships.
3. Add live-follow controls and terminal role reveal.
4. Add replay timeline, play/pause, speed, step and omniscient view.
5. Add decision trace inspection for completed games.

### Task 8: Training, agents and analytics

**Files:**
- Create: `web/src/pages/TrainingPage.tsx`
- Create: `web/src/pages/AgentsPage.tsx`
- Create: `web/src/pages/AnalyticsPage.tsx`
- Create: `web/src/pages/DashboardPage.tsx`

1. Implement typed job creation forms and parameter presets.
2. Add queue, live logs, cancellation, retry and artifact display.
3. Add Agent profile CRUD without secret persistence.
4. Add overview metrics, trends, role statistics and strategy distributions.
5. Link all analytical records back to games and jobs.

### Task 9: Verification and product delivery

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Add: frontend and backend tests as needed

1. Run targeted backend tests, then the complete Python suite.
2. Run Ruff, frontend unit tests, TypeScript checks and production build.
3. Launch the production-like service and execute browser end-to-end acceptance.
4. Verify responsive layouts and inspect browser console/network errors.
5. Document exact development and production commands and remaining experimental boundaries.

