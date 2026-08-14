# syntax=docker/dockerfile:1.7

FROM node:24-alpine AS web-builder
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.13-slim AS python-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy
WORKDIR /app
RUN pip install --no-cache-dir uv==0.11.7
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

FROM python-base AS studio-training
RUN uv sync --frozen --no-dev --extra train
COPY --from=web-builder /app/web/dist /app/web/dist
ENV WOLFPLAY_STUDIO_HOST=0.0.0.0 \
    WOLFPLAY_STUDIO_PORT=8000 \
    WOLFPLAY_STUDIO_DATA_DIR=/data \
    WOLFPLAY_STUDIO_ARTIFACT_DIR=/data/artifacts \
    WOLFPLAY_STUDIO_FRONTEND_DIST=/app/web/dist
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
  CMD ["uv", "run", "--no-sync", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"]
CMD ["uv", "run", "--no-sync", "wolfplay-web"]

FROM python-base AS studio
RUN uv sync --frozen --no-dev
COPY --from=web-builder /app/web/dist /app/web/dist
ENV WOLFPLAY_STUDIO_HOST=0.0.0.0 \
    WOLFPLAY_STUDIO_PORT=8000 \
    WOLFPLAY_STUDIO_DATA_DIR=/data \
    WOLFPLAY_STUDIO_ARTIFACT_DIR=/data/artifacts \
    WOLFPLAY_STUDIO_FRONTEND_DIST=/app/web/dist
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
  CMD ["uv", "run", "--no-sync", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"]
CMD ["uv", "run", "--no-sync", "wolfplay-web"]
