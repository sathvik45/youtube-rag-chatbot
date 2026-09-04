from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes.auth import router as auth_router
from src.api.routes.sources import router as sources_router
from src.api.routes.threads import router as threads_router

WEB_DIRECTORY = Path(__file__).resolve().parent / "web"

app = FastAPI(
    title="YouTube RAG API",
    version="0.1.0",
)

app.include_router(auth_router)
app.include_router(sources_router)
app.include_router(threads_router)



@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    """Serve the local browser UI from the same origin as the API."""
    return FileResponse(WEB_DIRECTORY / "index.html")


# Keep the API routes above this mount. Serving the UI from the API origin
# avoids a development CORS configuration and keeps provider configuration
# exclusively on the server.
app.mount("/ui", StaticFiles(directory=WEB_DIRECTORY), name="web-ui")
