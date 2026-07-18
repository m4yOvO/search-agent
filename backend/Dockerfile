FROM ghcr.io/astral-sh/uv:0.11.12 AS uv

FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=uv /uv /usr/local/bin/uv

COPY backend/pyproject.toml backend/uv.lock ./

# Install only the production dependency graph recorded in uv.lock. The application
# package itself is imported directly from /app, so no un-locked PEP 517 build
# environment is needed and this layer remains reusable when source files change.
RUN uv sync --frozen --no-dev --no-install-project

COPY backend/app ./app

# The three user-supplied JSON files are the only production mock-data source.
# Keep their original filenames and load them directly at runtime; no generated or
# curated fixture directory is copied into the image.
RUN mkdir -p /app/data
COPY ["data/person 1.json", "data/company 1.json", "data/relations 1.json", "/app/data/"]

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/runtime \
    && chown appuser:appuser /app/runtime

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
