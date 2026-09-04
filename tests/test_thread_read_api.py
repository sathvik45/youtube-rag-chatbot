import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.deps import (
    get_current_user,
    get_thread_history_reader,
    get_thread_list_reader,
)
from src.db.models.message import MessageRole, MessageStatus
from src.main import app
from src.services.auth import AuthenticatedUser
from src.services.threads import (
    ListedThread,
    ThreadHistory,
    ThreadHistoryMessage,
    ThreadMessageCitation,
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
        email="thread-read-api-test@example.invalid",
    )


def test_list_threads_returns_a_newest_active_page(
    client,
    authenticated_user,
) -> None:
    newest_updated_at = datetime(2026, 9, 4, 12, 30, tzinfo=timezone.utc)
    older_updated_at = newest_updated_at - timedelta(minutes=1)
    newest_thread = ListedThread(
        id=uuid.uuid4(),
        source_id=uuid.uuid4(),
        title="Newest chat",
        created_at=newest_updated_at - timedelta(hours=1),
        updated_at=newest_updated_at,
    )
    older_thread = ListedThread(
        id=uuid.uuid4(),
        source_id=uuid.uuid4(),
        title="Older chat",
        created_at=older_updated_at - timedelta(hours=1),
        updated_at=older_updated_at,
    )
    received_requests: list[
        tuple[uuid.UUID, datetime | None, uuid.UUID | None, int]
    ] = []

    def fake_read_threads(
        received_user_id: uuid.UUID,
        before_updated_at: datetime | None,
        before_id: uuid.UUID | None,
        received_limit: int,
    ) -> list[ListedThread]:
        received_requests.append(
            (
                received_user_id,
                before_updated_at,
                before_id,
                received_limit,
            )
        )
        return [newest_thread, older_thread]

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_list_reader] = (
        lambda: fake_read_threads
    )

    response = client.get("/threads?limit=1")

    assert response.status_code == 200

    body = response.json()
    assert body["items"] == [
        {
            "id": str(newest_thread.id),
            "source_id": str(newest_thread.source_id),
            "title": "Newest chat",
            "created_at": newest_thread.created_at.isoformat().replace(
                "+00:00", "Z"
            ),
            "updated_at": newest_thread.updated_at.isoformat().replace(
                "+00:00", "Z"
            ),
        }
    ]
    assert body["next_cursor"] is not None
    assert received_requests == [
        (authenticated_user.id, None, None, 2),
    ]


def test_list_threads_rejects_an_invalid_cursor_before_calling_reader(
    client,
    authenticated_user,
) -> None:
    def fake_read_threads(
        _: uuid.UUID,
        __: datetime | None,
        ___: uuid.UUID | None,
        ____: int,
    ) -> list[ListedThread]:
        raise AssertionError("The reader should not be called.")

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_list_reader] = (
        lambda: fake_read_threads
    )

    response = client.get("/threads?cursor=not-a-valid-cursor")

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid thread cursor."}


def test_list_threads_rejects_an_out_of_range_page_size(
    client,
    authenticated_user,
) -> None:
    def fake_read_threads(
        _: uuid.UUID,
        __: datetime | None,
        ___: uuid.UUID | None,
        ____: int,
    ) -> list[ListedThread]:
        raise AssertionError("The reader should not be called.")

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_list_reader] = (
        lambda: fake_read_threads
    )

    response = client.get("/threads?limit=101")

    assert response.status_code == 422


def test_get_thread_messages_returns_history_and_citations(
    client,
    authenticated_user,
) -> None:
    thread_id = uuid.uuid4()
    video_id = uuid.uuid4()
    user_message_id = uuid.uuid4()
    assistant_message_id = uuid.uuid4()
    citation_id = uuid.uuid4()
    user_created_at = datetime(2026, 9, 4, 9, 0, tzinfo=timezone.utc)
    assistant_created_at = user_created_at + timedelta(seconds=4)
    user_message = ThreadHistoryMessage(
        id=user_message_id,
        role=MessageRole.USER,
        content="What does this video teach?",
        rewritten_query="What does the uploaded video teach?",
        grounded=None,
        status=MessageStatus.USER_MESSAGE,
        created_at=user_created_at,
        citations=(),
    )
    assistant_message = ThreadHistoryMessage(
        id=assistant_message_id,
        role=MessageRole.ASSISTANT,
        content="It explains transaction boundaries.",
        rewritten_query=None,
        grounded=True,
        status=MessageStatus.ANSWERED,
        created_at=assistant_created_at,
        citations=(
            ThreadMessageCitation(
                id=citation_id,
                video_id=video_id,
                start_ms=12_000,
                end_ms=18_000,
                youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                quote_text="A transaction groups related changes.",
                verified=True,
                verification_score=0.92,
                position=1,
            ),
        ),
    )
    received_requests: list[
        tuple[
            uuid.UUID,
            uuid.UUID,
            datetime | None,
            uuid.UUID | None,
            int,
        ]
    ] = []

    def fake_read_history(
        received_user_id: uuid.UUID,
        received_thread_id: uuid.UUID,
        before_created_at: datetime | None,
        before_id: uuid.UUID | None,
        received_limit: int,
    ) -> ThreadHistory:
        received_requests.append(
            (
                received_user_id,
                received_thread_id,
                before_created_at,
                before_id,
                received_limit,
            )
        )
        return ThreadHistory(
            thread_id=received_thread_id,
            # The service returns raw newest-first pagination order.  The
            # endpoint reverses the visible page for a natural chat display.
            messages=(assistant_message, user_message),
        )

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_history_reader] = (
        lambda: fake_read_history
    )

    response = client.get(f"/threads/{thread_id}/messages?limit=2")

    assert response.status_code == 200
    assert response.json() == {
        "thread_id": str(thread_id),
        "items": [
            {
                "id": str(user_message_id),
                "role": "user",
                "content": "What does this video teach?",
                "rewritten_query": "What does the uploaded video teach?",
                "grounded": None,
                "status": "user_message",
                    "created_at": user_created_at.isoformat().replace(
                        "+00:00", "Z"
                    ),
                "citations": [],
            },
            {
                "id": str(assistant_message_id),
                "role": "assistant",
                "content": "It explains transaction boundaries.",
                "rewritten_query": None,
                "grounded": True,
                "status": "answered",
                    "created_at": assistant_created_at.isoformat().replace(
                        "+00:00", "Z"
                    ),
                "citations": [
                    {
                        "id": str(citation_id),
                        "video_id": str(video_id),
                        "start_ms": 12_000,
                        "end_ms": 18_000,
                        "youtube_url": (
                            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                        ),
                        "quote_text": (
                            "A transaction groups related changes."
                        ),
                        "verified": True,
                        "verification_score": 0.92,
                        "position": 1,
                    }
                ],
            },
        ],
        "next_cursor": None,
    }
    assert received_requests == [
        (authenticated_user.id, thread_id, None, None, 3),
    ]


def test_get_thread_messages_hides_unowned_or_missing_thread(
    client,
    authenticated_user,
) -> None:
    def fake_read_history(
        _: uuid.UUID,
        __: uuid.UUID,
        ___: datetime | None,
        ____: uuid.UUID | None,
        _____: int,
    ) -> ThreadHistory:
        raise LookupError("Thread was not found.")

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_history_reader] = (
        lambda: fake_read_history
    )

    response = client.get(f"/threads/{uuid.uuid4()}/messages")

    assert response.status_code == 404
    assert response.json() == {"detail": "Thread not found."}


def test_get_thread_messages_rejects_an_invalid_cursor_before_calling_reader(
    client,
    authenticated_user,
) -> None:
    def fake_read_history(
        _: uuid.UUID,
        __: uuid.UUID,
        ___: datetime | None,
        ____: uuid.UUID | None,
        _____: int,
    ) -> ThreadHistory:
        raise AssertionError("The reader should not be called.")

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_history_reader] = (
        lambda: fake_read_history
    )

    response = client.get(
        f"/threads/{uuid.uuid4()}/messages?cursor=not-a-valid-cursor"
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid message cursor."}


@pytest.mark.parametrize(
    "path",
    [
        "/threads",
        f"/threads/{uuid.uuid4()}/messages",
    ],
)
def test_thread_read_routes_require_a_bearer_token(client, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
