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
from src.db.models.source import  SourceType, SourceStatus, Source
from src.db.models.source_video import SourceVideo, SourceVideoStatus
from src.db.models.thread import Thread
from src.db.models.user import User
from src.db.models.video import TranscriptStatus, Video
from src.db.repositories.videos import get_or_create_video, update_video_status, mark_video_ready
from src.db.session import SessionLocal

from src.db.repositories.sources import (attach_videos, create_pending_source,refresh_source_status)


from src.db.models.ingestion_job import IngestionJobStatus, IngestionJob
from src.db.repositories.ingestion_jobs import (create_ingestion_job, mark_job_running, mark_job_succeeded)

from src.db.repositories.source_videos import update_source_video_status



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

        source = create_pending_source(
            session,
            user_id=user.id,
            source_type=SourceType.VIDEO,
            submitted_value=f"https://www.youtube.com/watch?v={suffix[:11]}",
        )       
        assert source.status is SourceStatus.PENDING

        video = get_or_create_video(
            session,
            youtube_video_id=suffix[:11],
            canonical_url=f"https://www.youtube.com/watch?v={suffix[:11]}",
            title="Database smoke-test video",
        )
        assert video.transcript_status == TranscriptStatus.PENDING

        reused_video = get_or_create_video(
            session,
            youtube_video_id=suffix[:11],
            canonical_url=f"https://www.youtube.com/watch?v={suffix[:11]}",
            title="a diff tittle should not create anoather row",
        )
        assert reused_video.id == video.id
        # session.add(video)
        # session.flush()

        source_videos =  attach_videos(
            session,
            source_id=source.id,
            video_ids= [video.id]
        )
        source_video = source_videos[0]

        assert source_video.position == 0
        assert source_video.status == SourceVideoStatus.PENDING

        job = create_ingestion_job(
            session,
            source_id=source.id,
            video_id=video.id
        )

        assert job.status == IngestionJobStatus.QUEUED
        assert job.source_id == source.id
        assert job.video_id == video.id

        running_job = mark_job_running(
            session,
            job_id=job.id
        )

        assert running_job.id == job.id
        assert running_job.status == IngestionJobStatus.RUNNING
        assert running_job.attempts == 1
        assert running_job.started_at is not None
        assert running_job.error_message is None

        processing_source_video = update_source_video_status(
            session,
            source_id= source.id,
            video_id= video.id,
            status=SourceVideoStatus.PROCESSING
        )

        assert processing_source_video.id == source_video.id
        assert processing_source_video.status == SourceVideoStatus.PROCESSING
        assert processing_source_video.error_message is None

        processing_video = update_video_status(
            session,
            video=video,
            status=TranscriptStatus.PROCESSING
        )

        assert processing_video.id == video.id
        assert processing_video.transcript_status == TranscriptStatus.PROCESSING
        assert processing_video.last_error is None


        processing_source = refresh_source_status(
            session,
            source_id=source.id,
        )

        assert processing_source.id == source.id
        assert processing_source.status == SourceStatus.PROCESSING

        ready_video = mark_video_ready(
            session,
            video,
            chunk_count=3,
            vector_count=3,
            title="Indexed database smoke-test video",
        )

        assert ready_video.transcript_status == TranscriptStatus.READY
        assert ready_video.chunk_count == 3
        assert ready_video.vector_count == 3

        ready_source_video = update_source_video_status(
            session,
            source_id=source.id,
            video_id=video.id,
            status=SourceVideoStatus.READY
        )
        assert ready_source_video.status == SourceVideoStatus.READY
        assert ready_source_video.error_message is None

        succeeded_job = mark_job_succeeded(
            session,
            job_id=job.id,
        )

        assert succeeded_job.status == IngestionJobStatus.SUCCEEDED
        assert succeeded_job.finished_at is not None

        ready_source = refresh_source_status(
            session,
            source_id=source.id,
        )
        assert ready_source.status == SourceStatus.READY


        thread = Thread(
            user_id=user.id,
            source_id=source.id,
            title="Database smoke-test thread",
        )
        session.add(thread)
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

        saved_video = session.scalar(
            select(Video).where(Video.id == video.id)
        )

        assert saved_video is not None
        assert saved_video.transcript_status == TranscriptStatus.READY
        assert saved_video.last_error is None
        assert saved_video.chunk_count == 3
        assert saved_video.vector_count == 3


        saved_source = session.scalar(
            select(Source).where(Source.id == source.id)
        )
        assert saved_source is not None
        assert saved_source.status == SourceStatus.READY

        saved_job = session.scalar(
            select(IngestionJob).where(IngestionJob.id == job.id)
        )

        assert saved_job is not None
        assert saved_job.status == IngestionJobStatus.SUCCEEDED
        assert saved_job.attempts == 1
        assert saved_job.started_at is not None
        assert saved_job.source_id == source.id 
        assert saved_job.video_id == video.id

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
        assert saved_source_video.status == SourceVideoStatus.READY
        completed_chain = True
    finally:
        transaction.rollback()
        if completed_chain:
            assert session.scalar(select(User.id).where(User.email == user.email)) is None
        session.close()
