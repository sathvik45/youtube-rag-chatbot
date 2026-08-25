"""Opt-in integration test for the core PostgreSQL persistence chain.

Run deliberately against a configured database with:
    $env:RUN_DB_INTEGRATION = "1"
    python -m pytest tests/test_db_smoke.py -q

Every row is flushed so foreign keys and queries are truly exercised, then the
outer transaction is rolled back in ``finally``. No test data is committed.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.db.models.citation import Citation
from src.db.models.message import Message, MessageRole, MessageStatus
from src.db.models.source import Source, SourceType
from src.db.models.source_video import SourceVideo, SourceVideoStatus
from src.db.models.thread import Thread
from src.db.models.user import User
from src.db.models.video import Video
from src.db.session import SessionLocal


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against the configured PostgreSQL database.",
)


def test_core_persistence_chain_rolls_back() -> None:
    """Create and read the full chat chain without committing it."""
    session = SessionLocal()
    transaction = session.begin()
    suffix = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc)

    completed_chain = False

    try:
        user = User(
            email=f"db-smoke-{suffix}@example.invalid",
            password_hash="test-only-not-a-real-password-hash",
        )
        session.add(user)
        session.flush()

        source = Source(
            user_id=user.id,
            source_type=SourceType.VIDEO,
            submitted_value=f"https://www.youtube.com/watch?v={suffix[:11]}",
        )
        video = Video(
            youtube_video_id=suffix[:11],
            canonical_url=f"https://www.youtube.com/watch?v={suffix[:11]}",
            title="Database smoke-test video",
        )
        session.add_all([source, video])
        session.flush()

        source_video = SourceVideo(
            source_id=source.id,
            video_id=video.id,
            position=0,
            status=SourceVideoStatus.READY,
        )
        thread = Thread(
            user_id=user.id,
            source_id=source.id,
            title="Database smoke-test thread",
        )
        session.add_all([source_video, thread])
        session.flush()

        user_message = Message(
            thread_id=thread.id,
            role=MessageRole.USER,
            content="What does the video cover?",
            rewritten_query="What topics does the video cover?",
            status=MessageStatus.USER_MESSAGE,
            created_at=created_at,
        )
        assistant_message = Message(
            thread_id=thread.id,
            role=MessageRole.ASSISTANT,
            content="It covers the database smoke-test topic.",
            grounded=True,
            status=MessageStatus.ANSWERED,
            created_at=created_at + timedelta(microseconds=1),
        )
        session.add_all([user_message, assistant_message])
        session.flush()

        citation = Citation(
            message_id=assistant_message.id,
            video_id=video.id,
            start_ms=1_000,
            end_ms=2_000,
            youtube_url=f"https://www.youtube.com/watch?v={video.youtube_video_id}&t=1s",
            quote_text="database smoke-test topic",
            verified=True,
            verification_score=1.0,
            position=1,
        )
        session.add(citation)
        session.flush()

        history = session.scalars(
            select(Message)
            .where(Message.thread_id == thread.id)
            .order_by(Message.created_at, Message.id)
        ).all()
        saved_citation = session.scalar(
            select(Citation).where(Citation.message_id == assistant_message.id)
        )
        saved_source_video = session.scalar(
            select(SourceVideo).where(SourceVideo.source_id == source.id)
        )

        assert [message.content for message in history] == [
            "What does the video cover?",
            "It covers the database smoke-test topic.",
        ]
        assert [message.role for message in history] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        assert saved_citation is not None
        assert saved_citation.video_id == video.id
        assert saved_citation.verified is True
        assert saved_citation.verification_score == 1.0
        assert saved_source_video is not None
        assert saved_source_video.video_id == video.id
        assert saved_source_video.status is SourceVideoStatus.READY
        completed_chain = True
    finally:
        transaction.rollback()
        if completed_chain:
            assert session.scalar(select(User.id).where(User.email == user.email)) is None
        session.close()
