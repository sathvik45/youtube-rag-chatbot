from fastapi import FastAPI

from src.api.routes.sources import router as sources_router
from src.api.routes.auth import router as auth_router

app = FastAPI(
    title="YouTube RAG API",
    version="0.1.0",
)

app.include_router(auth_router)
app.include_router(sources_router)



@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}