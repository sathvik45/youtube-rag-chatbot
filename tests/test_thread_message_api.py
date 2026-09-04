import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.deps import (
    get_current_user,
    get_thread_message_handler,
)
from src.db.models.message import MessageStatus
from src.main import app
from src.services.auth import AuthenticatedUser
from src.services.chat import (
    ChatAssistantMessage,
    ChatCitation,
    ChatProviderError,
    ChatTurn,
    ThreadMessageHandler,
    ThreadNotReadyForChatError,
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
        email="thread-message-api-test@example.invalid",
    )


def test_create_thread_message_returns_the_persisted_chat_turn(
    client,
    authenticated_user,
) -> None:
    thread_id = uuid.uuid4()
    user_message_id = uuid.uuid4()
    assistant_message_id = uuid.uuid4()
    citation_id = uuid.uuid4()
    video_id = uuid.uuid4()
    received_calls: list[tuple[uuid.UUID, uuid.UUID, str]] = []

    async def fake_handle_message(
        received_user_id: uuid.UUID,
        received_thread_id: uuid.UUID,
        content: str,
    ) -> ChatTurn:
        received_calls.append(
            (received_user_id, received_thread_id, content)
        )
        return ChatTurn(
            user_message_id=user_message_id,
            assistant_message=ChatAssistantMessage(
                id=assistant_message_id,
                content="The video explains database transactions.",
                status=MessageStatus.ANSWERED,
                grounded=True,
                citations=(
                    ChatCitation(
                        id=citation_id,
                        video_id=video_id,
                        start_ms=12_000,
                        end_ms=18_000,
                        youtube_url=(
                            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                        ),
                        quote_text="A transaction groups related changes.",
                        verified=True,
                        verification_score=0.92,
                        position=0,
                    ),
                ),
            ),
        )

    handler: ThreadMessageHandler = fake_handle_message
    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_message_handler] = lambda: handler

    response = client.post(
        f"/threads/{thread_id}/messages",
        json={"content": "  What does the video say about transactions?  "},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["user_message_id"] == str(user_message_id)

    assistant_message = body["assistant_message"]
    assert assistant_message["id"] == str(assistant_message_id)
    assert assistant_message["content"] == (
        "The video explains database transactions."
    )
    assert assistant_message["status"] == "answered"
    assert assistant_message["grounded"] is True
    assert assistant_message["citations"] == [
        {
            "id": str(citation_id),
            "video_id": str(video_id),
            "start_ms": 12_000,
            "end_ms": 18_000,
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "quote_text": "A transaction groups related changes.",
            "verified": True,
            "verification_score": 0.92,
            "position": 0,
        }
    ]
    assert received_calls == [
        (
            authenticated_user.id,
            thread_id,
            "What does the video say about transactions?",
        )
    ]


def test_create_thread_message_hides_unowned_or_missing_threads(
    client,
    authenticated_user,
) -> None:
    async def fake_handle_message(
        _: uuid.UUID,
        __: uuid.UUID,
        ___: str,
    ) -> ChatTurn:
        raise LookupError("Thread was not found.")

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_message_handler] = (
        lambda: fake_handle_message
    )

    response = client.post(
        f"/threads/{uuid.uuid4()}/messages",
        json={"content": "Can I read this private chat?"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Thread not found."}


def test_create_thread_message_rejects_a_thread_without_ready_videos(
    client,
    authenticated_user,
) -> None:
    async def fake_handle_message(
        _: uuid.UUID,
        __: uuid.UUID,
        ___: str,
    ) -> ChatTurn:
        raise ThreadNotReadyForChatError(
            "This thread does not have a ready video to chat with yet."
        )

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_message_handler] = (
        lambda: fake_handle_message
    )

    response = client.post(
        f"/threads/{uuid.uuid4()}/messages",
        json={"content": "Please answer now."},
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "This thread does not have a ready video to chat with yet."
    }


def test_create_thread_message_hides_provider_failures(
    client,
    authenticated_user,
) -> None:
    async def fake_handle_message(
        _: uuid.UUID,
        __: uuid.UUID,
        ___: str,
    ) -> ChatTurn:
        raise ChatProviderError("The vector database timed out.")

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_message_handler] = (
        lambda: fake_handle_message
    )

    response = client.post(
        f"/threads/{uuid.uuid4()}/messages",
        json={"content": "What is the main idea?"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Chat is temporarily unavailable. Please try again."
    }


def test_create_thread_message_rejects_blank_content_before_calling_service(
    client,
    authenticated_user,
) -> None:
    async def fake_handle_message(
        _: uuid.UUID,
        __: uuid.UUID,
        ___: str,
    ) -> ChatTurn:
        raise AssertionError("Service should not be called.")

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_message_handler] = (
        lambda: fake_handle_message
    )

    response = client.post(
        f"/threads/{uuid.uuid4()}/messages",
        json={"content": "   "},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "extra_field",
    [
        {"user_id": str(uuid.uuid4())},
        {"video_ids": [str(uuid.uuid4())]},
    ],
)
def test_create_thread_message_rejects_client_supplied_identity_or_scope(
    client,
    authenticated_user,
    extra_field: dict[str, object],
) -> None:
    async def fake_handle_message(
        _: uuid.UUID,
        __: uuid.UUID,
        ___: str,
    ) -> ChatTurn:
        raise AssertionError("Service should not be called.")

    app.dependency_overrides[get_current_user] = lambda: authenticated_user
    app.dependency_overrides[get_thread_message_handler] = (
        lambda: fake_handle_message
    )

    response = client.post(
        f"/threads/{uuid.uuid4()}/messages",
        json={
            "content": "What is in this thread's video?",
            **extra_field,
        },
    )

    assert response.status_code == 422


def test_thread_message_routes_require_a_bearer_token(client) -> None:
    response = client.post(
        f"/threads/{uuid.uuid4()}/messages",
        json={"content": "Unauthenticated question"},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
