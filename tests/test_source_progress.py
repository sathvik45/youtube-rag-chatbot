import os
import uuid

import pytest
from sqlalchemy.orm import sessionmaker

from src.db.models.ingestion_job import IngestionJob, IngestionJobStatus
from src.db.models.source import SourceStatus, SourceType
from src.db.models.source_video import SourceVideo, SourceVideoStatus
from src.db.models.user import User
from src.db.models.video import Video
from src.db.repositories.sources import (
    create_pending_source,
    get_source_progress,
    refresh_source_status,
)
from src.db.session import engine


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against the configured PostgreSQL database.",
)


@pytest.fixture
def transactional_session_factory():
    """Allow internal commits, then roll back all rows after the test."""
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


def test_get_source_progress_reports_video_and_job_counts(
    transactional_session_factory,
) -> None:
    suffix = uuid.uuid4().hex

    with transactional_session_factory() as session:
        with session.begin():
            user = User(
                email=f"progress-test-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()

            source = create_pending_source(
                session,
                user_id=user.id,
                source_type=SourceType.PLAYLIST,
                submitted_value="https://www.youtube.com/playlist?list=PL123456789",
                youtube_playlist_id="PL123456789",
            )

            videos = [
                Video(
                    youtube_video_id=f"{suffix[:10]}{index}",
                    canonical_url=(
                        f"https://www.youtube.com/watch?v={suffix[:10]}{index}"
                    ),
                    title=f"Progress test video {index}",
                )
                for index in range(3)
            ]
            session.add_all(videos)
            session.flush()

            source_videos = [
                SourceVideo(
                    source_id=source.id,
                    video_id=videos[0].id,
                    position=0,
                    status=SourceVideoStatus.READY,
                ),
                SourceVideo(
                    source_id=source.id,
                    video_id=videos[1].id,
                    position=1,
                    status=SourceVideoStatus.PROCESSING,
                ),
                SourceVideo(
                    source_id=source.id,
                    video_id=videos[2].id,
                    position=2,
                    status=SourceVideoStatus.NO_TRANSCRIPT,
                ),
            ]
            session.add_all(source_videos)

            jobs = [
                IngestionJob(
                    source_id=source.id,
                    video_id=videos[0].id,
                    status=IngestionJobStatus.SUCCEEDED,
                ),
                IngestionJob(
                    source_id=source.id,
                    video_id=videos[1].id,
                    status=IngestionJobStatus.RUNNING,
                ),
                IngestionJob(
                    source_id=source.id,
                    video_id=videos[2].id,
                    status=IngestionJobStatus.RETRYING,
                ),
            ]
            session.add_all(jobs)
            session.flush()

            refresh_source_status(
                session,
                source_id=source.id,
            )

            source_id = source.id

    with transactional_session_factory() as session:
        progress = get_source_progress(
            session,
            source_id=source_id,
        )

        assert progress.source_id == source_id
        assert progress.source_status == SourceStatus.PROCESSING

        assert progress.video_counts["pending"] == 0
        assert progress.video_counts["processing"] == 1
        assert progress.video_counts["ready"] == 1
        assert progress.video_counts["no_transcript"] == 1
        assert progress.video_counts["failed"] == 0

        assert progress.job_counts["queued"] == 0
        assert progress.job_counts["running"] == 1
        assert progress.job_counts["succeeded"] == 1
        assert progress.job_counts["retrying"] == 1
        assert progress.job_counts["failed"] == 0

        assert progress.total_videos == 3
        assert progress.completed_videos == 2