"""Application package root.

Everything imports as `src.<subpackage>.<module>` and every entrypoint runs
from the repo root (`python -m src.rag.ingest`, `uvicorn src.main:app`).

Why this file exists: the package previously had no `__init__.py` anywhere and
two competing conventions -- `src/rag/*` and `src/core/*` imported each other
as `core.config`, while `src/Graph/*` imported `src.core.config`. Those need
different sys.path roots (`src/` vs the repo root), so they can never both
resolve inside one process. A FastAPI app is one process, so this had to be
settled before anything else could be built on top of it.
"""
