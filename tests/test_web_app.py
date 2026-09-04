import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    app.dependency_overrides.clear()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_root_serves_the_local_browser_app(client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "YouTube RAG" in response.text
    assert '/ui/styles.css' in response.text
    assert '/ui/app.js' in response.text


@pytest.mark.parametrize(
    "path",
    [
        "/ui/styles.css",
        "/ui/app.js",
        "/health",
    ],
)
def test_browser_assets_and_health_route_are_available(client, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
