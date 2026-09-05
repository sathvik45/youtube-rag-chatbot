"""Route-level contract tests for permanent, owner-scoped chat deletion."""

import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_current_user, get_thread_deleter
from src.main import app
from src.services.auth import AuthenticatedUser


@pytest.fixture
def client():
    app.dependency_overrides.clear()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def authenticated_user() -> AuthenticatedUser:
    return AuthenticatedUser(
        id=uuid.uuid4(),
        email="thread-delete-api-test@example.invalid",
    )


def test_delete_owned_thread_returns_no_content_and_uses_current_user(
    client,
    authenticated_user,
) -> None:
    thread_id = uuid.uuid4()
    received_calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    def fake_delete_thread(
        received_user_id: uuid.UUID,
        received_thread_id: uuid.UUID,
    ) -> None:
        received_calls.append((received_user_id, received_thread_id))

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_deleter] = lambda: fake_delete_thread

    response = client.delete(f"/threads/{thread_id}")

    assert response.status_code == 204
    assert response.content == b""
    assert received_calls == [(authenticated_user.id, thread_id)]


@pytest.mark.parametrize(
    "lookup_failure",
    [
        "Thread was not found.",
        "Thread belongs to a different user.",
    ],
    ids=["missing", "unowned"],
)
def test_delete_thread_hides_missing_or_unowned_threads(
    client,
    authenticated_user,
    lookup_failure: str,
) -> None:
    def fake_delete_thread(
        _: uuid.UUID,
        __: uuid.UUID,
    ) -> None:
        raise LookupError(lookup_failure)

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_deleter] = lambda: fake_delete_thread

    response = client.delete(f"/threads/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Thread not found."}


def test_delete_thread_requires_a_bearer_token(client) -> None:
    response = client.delete(f"/threads/{uuid.uuid4()}")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
