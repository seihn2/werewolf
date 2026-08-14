.PHONY: install install-train dev-api dev-web build serve test lint clean

install:
	uv sync --extra dev
	cd web && npm install

install-train:
	uv sync --extra dev --extra train
	cd web && npm install

dev-api:
	uv run uvicorn wolfplay.web.app:create_app --factory --reload --host 127.0.0.1 --port 8000

dev-web:
	cd web && npm run dev

build:
	cd web && npm run build

serve: build
	uv run wolfplay-web

test:
	uv run pytest
	cd web && npm test

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	cd web && npm run typecheck && npm run lint

clean:
	rm -rf web/dist .pytest_cache .ruff_cache
