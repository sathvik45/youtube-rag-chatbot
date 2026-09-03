import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.deps import (
    get_current_user,
    get_thread_creator,
)
from src.main import app
from src.services.auth import AuthenticatedUser
from src.services.threads import (
    CreatedThread,
    SourceNotReadyForChatError,
)


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
        email="thread-api-test@example.invalid",
    )


def test_create_thread_returns_a_source_scoped_chat(
    client,
    authenticated_user,
) -> None:
    source_id = uuid.uuid4()
    thread_id = uuid.uuid4()
    created_at = datetime(2026, 9, 3, tzinfo=timezone.utc)
    received_calls: list[tuple[uuid.UUID, uuid.UUID, str]] = []

    def fake_create_thread(
        user_id: uuid.UUID,
        received_source_id: uuid.UUID,
        title: str,
    ) -> CreatedThread:
        received_calls.append((user_id, received_source_id, title))
        return CreatedThread(
            id=thread_id,
            source_id=received_source_id,
            title=title,
            created_at=created_at,
        )

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_creator] = lambda: fake_create_thread

    response = client.post(
        "/threads",
        json={
            "source_id": str(source_id),
            "title": "  Study notes  ",
        },
    )

    assert response.status_code == 201

    body = response.json()
    assert body["id"] == str(thread_id)
    assert body["source_id"] == str(source_id)
    assert body["title"] == "Study notes"
    assert datetime.fromisoformat(body["created_at"]) == created_at
    assert received_calls == [
        (authenticated_user.id, source_id, "Study notes"),
    ]


def test_create_thread_hides_unowned_or_missing_sources(
    client,
    authenticated_user,
) -> None:
    source_id = uuid.uuid4()

    def fake_create_thread(
        _: uuid.UUID,
        __: uuid.UUID,
        ___: str,
    ) -> CreatedThread:
        raise LookupError("Source was not found.")

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_creator] = lambda: fake_create_thread

    response = client.post(
        "/threads",
        json={
            "source_id": str(source_id),
            "title": "Private source",
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Source not found."}


def test_create_thread_rejects_a_source_that_is_not_chat_ready(
    client,
    authenticated_user,
) -> None:
    source_id = uuid.uuid4()

    def fake_create_thread(
        _: uuid.UUID,
        __: uuid.UUID,
        ___: str,
    ) -> CreatedThread:
        raise SourceNotReadyForChatError(
            "Source is not ready for chat yet."
        )

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_creator] = lambda: fake_create_thread

    response = client.post(
        "/threads",
        json={
            "source_id": str(source_id),
            "title": "Waiting for ingestion",
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Source is not ready for chat yet.",
    }


def test_create_thread_rejects_a_blank_title_before_calling_service(
    client,
    authenticated_user,
) -> None:
    def fake_create_thread(
        _: uuid.UUID,
        __: uuid.UUID,
        ___: str,
    ) -> CreatedThread:
        raise AssertionError("Service should not be called.")

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_creator] = lambda: fake_create_thread

    response = client.post(
        "/threads",
        json={
            "source_id": str(uuid.uuid4()),
            "title": "   ",
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "extra_field",
    [
        {"user_id": str(uuid.uuid4())},
        {"video_ids": [str(uuid.uuid4())]},
    ],
)
def test_create_thread_rejects_client_supplied_identity_or_scope(
    client,
    authenticated_user,
    extra_field: dict[str, object],
) -> None:
    def fake_create_thread(
        _: uuid.UUID,
        __: uuid.UUID,
        ___: str,
    ) -> CreatedThread:
        raise AssertionError("Service should not be called.")

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_creator] = lambda: fake_create_thread

    response = client.post(
        "/threads",
        json={
            "source_id": str(uuid.uuid4()),
            "title": "Scoped chat",
            **extra_field,
        },
    )

    assert response.status_code == 422


def test_thread_routes_require_a_bearer_token(client) -> None:
    response = client.post(
        "/threads",
        json={
            "source_id": str(uuid.uuid4()),
            "title": "Unauthenticated",
        },
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
