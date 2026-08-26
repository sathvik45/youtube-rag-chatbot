import os
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from src.db.models.ingestion_job import IngestionJob, IngestionJobStatus
from src.db.models.source import Source, SourceStatus, SourceType
from src.db.models.source_video import SourceVideo, SourceVideoStatus
from src.db.models.user import User
from src.db.models.video import TranscriptStatus, Video
from src.db.session import engine
from src.services.source_submission import submit_source


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against the configured PostgreSQL database.",
)


@pytest.fixture
def transactional_session_factory():
    """Allow service commits, then roll them all back after the test."""
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


def test_submit_playlist_creates_videos_links_and_jobs(
    transactional_session_factory,
) -> None:
    suffix = uuid.uuid4().hex
    submitted_url = "https://www.youtube.com/playlist?list=PL123456789"
    resolved_video_ids = [
        "dQw4w9WgXcQ",
        "9bZkp7q19f0",
    ]

    with transactional_session_factory() as session:
        with session.begin():
            user = User(
                email=f"submission-test-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()

            user_id = user.id

    def fake_resolve(received_url: str) -> list[str]:
        assert received_url == submitted_url
        return resolved_video_ids

    submission = submit_source(
        user_id,
        submitted_url,
        resolve=fake_resolve,
        session_factory=transactional_session_factory,
    )

    assert len(submission.video_ids) == 2
    assert len(submission.job_ids) == 2

    with transactional_session_factory() as session:
        source = session.get(Source, submission.source_id)
        videos = session.scalars(
            select(Video).where(Video.id.in_(submission.video_ids))
        ).all()
        source_videos = session.scalars(
            select(SourceVideo)
            .where(SourceVideo.source_id == submission.source_id)
            .order_by(SourceVideo.position)
        ).all()
        jobs = session.scalars(
            select(IngestionJob).where(
                IngestionJob.source_id == submission.source_id
            )
        ).all()

        assert source is not None
        assert source.user_id == user_id
        assert source.source_type == SourceType.PLAYLIST
        assert source.youtube_playlist_id == "PL123456789"
        assert source.status == SourceStatus.PROCESSING

        assert {video.youtube_video_id for video in videos} == set(
            resolved_video_ids
        )
        assert all(
            video.transcript_status == TranscriptStatus.PENDING
            for video in videos
        )

        assert [source_video.position for source_video in source_videos] == [0, 1]
        assert all(
            source_video.status == SourceVideoStatus.PENDING
            for source_video in source_videos
        )
        assert {source_video.video_id for source_video in source_videos} == set(
            submission.video_ids
        )

        assert {job.id for job in jobs} == set(submission.job_ids)
        assert all(job.status == IngestionJobStatus.QUEUED for job in jobs)
        assert {job.video_id for job in jobs} == set(submission.video_ids)