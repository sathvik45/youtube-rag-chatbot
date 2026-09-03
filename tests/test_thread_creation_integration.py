import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from src.api.deps import get_session_factory
from src.core.security import create_access_token
from src.db.models.source import Source, SourceStatus, SourceType
from src.db.models.source_video import SourceVideoStatus
from src.db.models.thread import Thread
from src.db.models.user import User
from src.db.repositories.sources import attach_videos, create_pending_source
from src.db.repositories.videos import get_or_create_video
from src.db.session import engine
from src.main import app


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against PostgreSQL.",
)


@pytest.fixture
def transactional_session_factory():
    """Allow request transactions while rolling back all test data at the end."""
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
    app.dependency_overrides.clear()
    app.dependency_overrides[get_session_factory] = (
        lambda: transactional_session_factory
    )

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def _create_source(
    session: Session,
    *,
    user_id: uuid.UUID,
    youtube_video_id: str,
    status: SourceStatus,
) -> Source:
    source = create_pending_source(
        session,
        user_id=user_id,
        source_type=SourceType.VIDEO,
        submitted_value=(
            f"https://www.youtube.com/watch?v={youtube_video_id}"
        ),
    )

    if status in {SourceStatus.READY, SourceStatus.PARTIAL}:
        video = get_or_create_video(
            session,
            youtube_video_id=youtube_video_id,
            canonical_url=(
                f"https://www.youtube.com/watch?v={youtube_video_id}"
            ),
            title="Ready thread source",
        )
        source_video = attach_videos(
            session,
            source_id=source.id,
            video_ids=[video.id],
        )[0]
        source_video.status = SourceVideoStatus.READY

    source.status = status
    session.flush()

    return source


@pytest.mark.parametrize(
    "source_status",
    [SourceStatus.READY, SourceStatus.PARTIAL],
)
def test_authenticated_user_can_create_a_thread_for_a_chat_ready_source(
    transactional_session_factory,
    transactional_client,
    source_status: SourceStatus,
) -> None:
    suffix = uuid.uuid4().hex
    youtube_video_id = suffix[:11]
    title = "Study notes"

    with transactional_session_factory() as session:
        with session.begin():
            user = User(
                email=f"thread-owner-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()

            source = _create_source(
                session,
                user_id=user.id,
                youtube_video_id=youtube_video_id,
                status=source_status,
            )

            user_id = user.id
            source_id = source.id

    response = transactional_client.post(
        "/threads",
        headers={"Authorization": f"Bearer {create_access_token(user_id)}"},
        json={"source_id": str(source_id), "title": title},
    )

    assert response.status_code == 201
    thread_id = uuid.UUID(response.json()["id"])

    with transactional_session_factory() as session:
        thread = session.get(Thread, thread_id)

    assert thread is not None
    assert thread.user_id == user_id
    assert thread.source_id == source_id
    assert thread.title == title


def test_user_cannot_create_a_thread_for_another_users_source(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex

    with transactional_session_factory() as session:
        with session.begin():
            owner = User(
                email=f"thread-owner-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            other_user = User(
                email=f"thread-other-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add_all([owner, other_user])
            session.flush()

            source = _create_source(
                session,
                user_id=owner.id,
                youtube_video_id=suffix[:11],
                status=SourceStatus.READY,
            )

            other_user_id = other_user.id
            source_id = source.id

    response = transactional_client.post(
        "/threads",
        headers={
            "Authorization": f"Bearer {create_access_token(other_user_id)}"
        },
        json={"source_id": str(source_id), "title": "Not mine"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Source not found."}


def test_user_cannot_create_a_thread_before_source_ingestion_is_ready(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex

    with transactional_session_factory() as session:
        with session.begin():
            user = User(
                email=f"thread-pending-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()

            source = _create_source(
                session,
                user_id=user.id,
                youtube_video_id=suffix[:11],
                status=SourceStatus.PROCESSING,
            )

            user_id = user.id
            source_id = source.id

    response = transactional_client.post(
        "/threads",
        headers={"Authorization": f"Bearer {create_access_token(user_id)}"},
        json={"source_id": str(source_id), "title": "Too early"},
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Source is not ready for chat yet.",
    }


def test_user_cannot_create_a_thread_when_ready_source_has_no_ready_videos(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex

    with transactional_session_factory() as session:
        with session.begin():
            user = User(
                email=f"thread-empty-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()

            source = create_pending_source(
                session,
                user_id=user.id,
                source_type=SourceType.VIDEO,
                submitted_value=(
                    f"https://www.youtube.com/watch?v={suffix[:11]}"
                ),
            )
            source.status = SourceStatus.READY

            user_id = user.id
            source_id = source.id

    response = transactional_client.post(
        "/threads",
        headers={"Authorization": f"Bearer {create_access_token(user_id)}"},
        json={"source_id": str(source_id), "title": "No ready videos"},
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Source has no ready videos to use for chat.",
    }
