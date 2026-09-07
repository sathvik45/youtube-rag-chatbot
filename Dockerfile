# Production image for the FastAPI service. The same image can later run the
# ingestion worker by overriding this default command in a Cloud Run Job.
FROM python:3.11-slim

# Copy the locked uv package manager without adding an installer or curl to the
# final image. Keep this version within the project's uv_build 0.12.x range.
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:${PATH}" \
    UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1 \
    PORT=8080 \
    DATA_DIR=/tmp/rag-data \
    HF_HOME=/tmp/huggingface

WORKDIR /app

# sentence-transformers/PyTorch uses the GNU OpenMP runtime. We intentionally
# avoid compilers and other build tools unless a future image build proves they
# are necessary.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --create-home app

# This dependency layer changes only when dependency metadata changes. A source
# edit therefore reuses it instead of downloading and installing packages again.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

# Copy only the runtime material permitted by .dockerignore. The current
# project entrypoint is imported directly from /app rather than installed as a
# package, so that the image uses the same src.* imports as local development.
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app src ./src

USER app

EXPOSE 8080

# Cloud Run injects PORT at runtime. The shell performs that substitution;
# exec forwards shutdown signals cleanly to Uvicorn.
CMD ["sh", "-c", "exec uvicorn src.main:app --host 0.0.0.0 --port \"${PORT:-8080}\""]
