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


def test_browser_app_renders_assistant_markdown_with_safe_dom_nodes(client) -> None:
    script = client.get("/ui/app.js").text
    stylesheet = client.get("/ui/styles.css").text

    assert "function renderAssistantMarkdown" in script
    assert "content.append(renderAssistantMarkdown(message.content));" in script
    assert "document.createElement(\"strong\")" in script
    assert "document.createElement(\"a\")" in script
    assert "url.protocol === \"https:\"" in script
    assert "content.innerHTML" not in script
    assert "insertAdjacentHTML" not in script
    assert ".message.assistant .message-content pre" in stylesheet


def test_browser_app_supports_permanent_chat_deletion(client) -> None:
    page = client.get("/").text
    script = client.get("/ui/app.js").text
    stylesheet = client.get("/ui/styles.css").text

    assert 'row.className = "thread-row";' in script
    assert 'method: "DELETE"' in script
    assert "window.confirm(" in script
    assert "The uploaded source and its vector data will not be deleted." in script
    assert "Chat deleted. Its uploaded source and vector data were kept." in script
    assert 'id="toggle-thread-view"' not in page
    assert "threadView" not in script
    assert "archived" not in script
    assert ".thread-delete-button" in stylesheet
