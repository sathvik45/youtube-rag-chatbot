"""Opt-in PostgreSQL coverage for permanent thread deletion.

The test uses one outer transaction, so it exercises real foreign-key
cascades without retaining any seeded or API-created data after the test.
"""

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from src.api.deps import get_session_factory
from src.core.security import create_access_token
from src.db.models.citation import Citation
from src.db.models.message import Message, MessageRole, MessageStatus
from src.db.models.source import Source, SourceStatus, SourceType
from src.db.models.source_video import SourceVideo, SourceVideoStatus
from src.db.models.thread import Thread
from src.db.models.user import User
from src.db.models.video import TranscriptStatus, Video
from src.db.repositories.sources import attach_videos, create_pending_source
from src.db.repositories.videos import get_or_create_video
from src.db.session import engine, get_db
from src.main import app
from src.rag import vector_store


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against PostgreSQL.",
)


@pytest.fixture
def transactional_session_factory():
    """Allow request transactions while rolling all test rows back."""
    connection = engine.connect()
    outer_transaction = connection.begin()

    factory = sessionmaker(
        bind=connection,
        autoflush=False,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    try:
        yield factory
    finally:
        outer_transaction.rollback()
        connection.close()


@pytest.fixture
def transactional_client(transactional_session_factory):
    """Make auth and deletion use the rollback-bound session factory."""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_session_factory] = (
        lambda: transactional_session_factory
    )

    def override_get_db():
        with transactional_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def _seed_thread_with_history(
    session: Session,
    *,
    suffix: str,
) -> tuple[
    User,
    Source,
    Video,
    SourceVideo,
    Thread,
    Message,
    Message,
    Citation,
]:
    """Create exactly the rows that must and must not survive deletion."""
    owner = User(
        email=f"thread-delete-owner-{suffix}@example.invalid",
        password_hash="test-only-not-a-real-password-hash",
    )
    session.add(owner)
    session.flush()

    youtube_video_id = suffix[:11]
    source = create_pending_source(
        session,
        user_id=owner.id,
        source_type=SourceType.VIDEO,
        submitted_value=(
            f"https://www.youtube.com/watch?v={youtube_video_id}"
        ),
    )
    source.status = SourceStatus.READY

    video = get_or_create_video(
        session,
        youtube_video_id=youtube_video_id,
        canonical_url=(
            f"https://www.youtube.com/watch?v={youtube_video_id}"
        ),
        title="Deletion-preserved video",
    )
    video.transcript_status = TranscriptStatus.READY
    video.chunk_count = 3
    video.vector_count = 3

    source_video = attach_videos(
        session,
        source_id=source.id,
        video_ids=[video.id],
    )[0]
    source_video.status = SourceVideoStatus.READY

    thread = Thread(
        user_id=owner.id,
        source_id=source.id,
        title="Chat to permanently delete",
    )
    session.add(thread)
    session.flush()

    user_message = Message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="What does this video explain?",
        rewritten_query="What does the uploaded video explain?",
        status=MessageStatus.USER_MESSAGE,
    )
    assistant_message = Message(
        thread_id=thread.id,
        role=MessageRole.ASSISTANT,
        content="It explains thread-scoped deletion.",
        grounded=True,
        status=MessageStatus.ANSWERED,
    )
    session.add_all([user_message, assistant_message])
    session.flush()

    citation = Citation(
        message_id=assistant_message.id,
        video_id=video.id,
        start_ms=1_000,
        end_ms=4_000,
        youtube_url=f"{video.canonical_url}&t=1s",
        quote_text="thread-scoped deletion",
        verified=True,
        verification_score=0.99,
        position=1,
    )
    session.add(citation)
    session.flush()

    return (
        owner,
        source,
        video,
        source_video,
        thread,
        user_message,
        assistant_message,
        citation,
    )


def test_delete_thread_removes_chat_history_but_preserves_ingested_content(
    transactional_session_factory,
    transactional_client,
    monkeypatch,
) -> None:
    """Only chat-owned data is deleted; vectors and source data stay intact."""
    vector_delete_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def unexpected_vector_delete(
        *args: object,
        **kwargs: object,
    ) -> int:
        vector_delete_calls.append((args, kwargs))
        raise AssertionError("Deleting a chat must not delete video vectors.")

    # Keep this integration test offline even if a future implementation
    # accidentally starts reaching for Pinecone while deleting a chat.
    monkeypatch.setattr(
        vector_store,
        "delete_video",
        unexpected_vector_delete,
    )
    monkeypatch.setattr(
        "src.services.threads.delete_video",
        unexpected_vector_delete,
        raising=False,
    )

    suffix = uuid.uuid4().hex
    with transactional_session_factory() as session:
        with session.begin():
            (
                owner,
                source,
                video,
                source_video,
                thread,
                user_message,
                assistant_message,
                citation,
            ) = _seed_thread_with_history(session, suffix=suffix)
            owner_id = owner.id
            source_id = source.id
            video_id = video.id
            source_video_id = source_video.id
            thread_id = thread.id
            user_message_id = user_message.id
            assistant_message_id = assistant_message.id
            citation_id = citation.id

    response = transactional_client.delete(
        f"/threads/{thread_id}",
        headers={"Authorization": f"Bearer {create_access_token(owner_id)}"},
    )

    assert response.status_code == 204
    assert response.content == b""
    assert vector_delete_calls == []

    history_response = transactional_client.get(
        f"/threads/{thread_id}/messages",
        headers={"Authorization": f"Bearer {create_access_token(owner_id)}"},
    )
    assert history_response.status_code == 404
    assert history_response.json() == {"detail": "Thread not found."}

    message_response = transactional_client.post(
        f"/threads/{thread_id}/messages",
        headers={"Authorization": f"Bearer {create_access_token(owner_id)}"},
        json={"content": "Can this deleted chat answer one more question?"},
    )
    assert message_response.status_code == 404
    assert message_response.json() == {"detail": "Thread not found."}

    with transactional_session_factory() as session:
        assert session.get(Thread, thread_id) is None
        assert session.get(Message, user_message_id) is None
        assert session.get(Message, assistant_message_id) is None
        assert session.get(Citation, citation_id) is None

        preserved_source = session.get(Source, source_id)
        preserved_video = session.get(Video, video_id)
        preserved_source_video = session.get(SourceVideo, source_video_id)

        assert preserved_source is not None
        assert preserved_source.status is SourceStatus.READY
        assert preserved_video is not None
        assert preserved_video.transcript_status is TranscriptStatus.READY
        assert preserved_video.chunk_count == 3
        assert preserved_video.vector_count == 3
        assert preserved_source_video is not None
        assert preserved_source_video.source_id == source_id
        assert preserved_source_video.video_id == video_id
        assert preserved_source_video.status is SourceVideoStatus.READY


def test_delete_thread_hides_another_users_chat_and_leaves_it_untouched(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex
    with transactional_session_factory() as session:
        with session.begin():
            (
                owner,
                _,
                _,
                _,
                thread,
                user_message,
                assistant_message,
                citation,
            ) = _seed_thread_with_history(session, suffix=suffix)
            other_user = User(
                email=(
                    f"thread-delete-other-{suffix}@example.invalid"
                ),
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(other_user)
            session.flush()

            owner_id = owner.id
            other_user_id = other_user.id
            thread_id = thread.id
            user_message_id = user_message.id
            assistant_message_id = assistant_message.id
            citation_id = citation.id

    response = transactional_client.delete(
        f"/threads/{thread_id}",
        headers={
            "Authorization": f"Bearer {create_access_token(other_user_id)}"
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Thread not found."}

    with transactional_session_factory() as session:
        saved_thread = session.get(Thread, thread_id)
        assert saved_thread is not None
        assert saved_thread.user_id == owner_id
        assert session.get(Message, user_message_id) is not None
        assert session.get(Message, assistant_message_id) is not None
        assert session.get(Citation, citation_id) is not None


def test_delete_missing_thread_returns_generic_not_found(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex
    with transactional_session_factory() as session:
        with session.begin():
            owner = User(
                email=f"thread-delete-missing-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(owner)
            session.flush()
            owner_id = owner.id

    response = transactional_client.delete(
        f"/threads/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {create_access_token(owner_id)}"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Thread not found."}
