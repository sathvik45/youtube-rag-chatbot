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
from src.db.repositories.ingestion_jobs import create_ingestion_job
from src.db.repositories.sources import attach_videos, create_pending_source
from src.db.repositories.videos import get_or_create_video
from src.db.session import engine
from src.rag.ingest import IngestResult, VideoStatus
from src.services.ingestion_worker import run_ingestion_job


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against the configured PostgreSQL database.",
)


@pytest.fixture
def transactional_session_factory():
    """Let worker transactions commit, then discard everything at test end."""
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


def test_worker_records_successful_ingestion(transactional_session_factory) -> None:
    suffix = uuid.uuid4().hex
    youtube_video_id = suffix[:11]

    with transactional_session_factory() as session:
        with session.begin():
            user = User(
                email=f"worker-test-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()

            source = create_pending_source(
                session,
                user_id=user.id,
                source_type=SourceType.VIDEO,
                submitted_value=f"https://www.youtube.com/watch?v={youtube_video_id}",
            )
            video = get_or_create_video(
                session,
                youtube_video_id=youtube_video_id,
                canonical_url=f"https://www.youtube.com/watch?v={youtube_video_id}",
                title="Worker test video",
            )
            attach_videos(
                session,
                source_id=source.id,
                video_ids=[video.id],
            )
            job = create_ingestion_job(
                session,
                source_id=source.id,
                video_id=video.id,
            )

            job_id = job.id
            source_id = source.id
            video_id = video.id

    def fake_ingest(received_video_id: str) -> IngestResult:
        assert received_video_id == youtube_video_id

        return IngestResult(
            video_id=received_video_id,
            status=VideoStatus.OK,
            chunks=3,
            vectors=3,
            title="Indexed worker test video",
        )

    result = run_ingestion_job(
        job_id,
        ingest=fake_ingest,
        session_factory=transactional_session_factory,
    )

    assert result.status == VideoStatus.OK

    with transactional_session_factory() as session:
        job = session.get(IngestionJob, job_id)
        source = session.get(Source, source_id)
        video = session.get(Video, video_id)
        source_video = session.scalar(
            select(SourceVideo).where(
                SourceVideo.source_id == source_id,
                SourceVideo.video_id == video_id,
            )
        )

        assert job is not None
        assert job.status == IngestionJobStatus.SUCCEEDED
        assert job.attempts == 1
        assert job.started_at is not None
        assert job.finished_at is not None

        assert source_video is not None
        assert source_video.status == SourceVideoStatus.READY

        assert video is not None
        assert video.transcript_status == TranscriptStatus.READY
        assert video.chunk_count == 3
        assert video.vector_count == 3
        assert video.title == "Indexed worker test video"

        assert source is not None
        assert source.status == SourceStatus.READY

@pytest.mark.parametrize(
    (
        "result_status",
        "expected_job_status",
        "expected_source_video_status",
        "expected_video_status",
    ),
    [
        (
            VideoStatus.NO_TRANSCRIPT,
            IngestionJobStatus.SUCCEEDED,
            SourceVideoStatus.NO_TRANSCRIPT,
            TranscriptStatus.NO_TRANSCRIPT,
        ),
        (
            VideoStatus.ERROR,
            IngestionJobStatus.RETRYING,
            SourceVideoStatus.FAILED,
            TranscriptStatus.FAILED,
        ),
    ],
)
def test_worker_records_non_successful_ingestion(
    transactional_session_factory,
    result_status: VideoStatus,
    expected_job_status: IngestionJobStatus,
    expected_source_video_status: SourceVideoStatus,
    expected_video_status: TranscriptStatus,
) -> None:
    suffix = uuid.uuid4().hex
    youtube_video_id = suffix[:11]
    error_message = "Test ingestion failure."

    with transactional_session_factory() as session:
        with session.begin():
            user = User(
                email=f"worker-error-test-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()

            source = create_pending_source(
                session,
                user_id=user.id,
                source_type=SourceType.VIDEO,
                submitted_value=f"https://www.youtube.com/watch?v={youtube_video_id}",
            )
            video = get_or_create_video(
                session,
                youtube_video_id=youtube_video_id,
                canonical_url=f"https://www.youtube.com/watch?v={youtube_video_id}",
                title="Worker error test video",
            )
            attach_videos(
                session,
                source_id=source.id,
                video_ids=[video.id],
            )
            job = create_ingestion_job(
                session,
                source_id=source.id,
                video_id=video.id,
            )

            job_id = job.id
            source_id = source.id
            video_id = video.id

    def fake_ingest(received_video_id: str) -> IngestResult:
        assert received_video_id == youtube_video_id

        return IngestResult(
            video_id=received_video_id,
            status=result_status,
            error="test-error",
            message=error_message,
        )

    result = run_ingestion_job(
        job_id,
        ingest=fake_ingest,
        session_factory=transactional_session_factory,
    )

    assert result.status == result_status

    with transactional_session_factory() as session:
        job = session.get(IngestionJob, job_id)
        source = session.get(Source, source_id)
        video = session.get(Video, video_id)
        source_video = session.scalar(
            select(SourceVideo).where(
                SourceVideo.source_id == source_id,
                SourceVideo.video_id == video_id,
            )
        )

        assert job is not None
        assert job.status == expected_job_status
        assert job.attempts == 1
        assert job.finished_at is not None

        assert source_video is not None
        assert source_video.status == expected_source_video_status
        assert source_video.error_message == error_message

        assert video is not None
        assert video.transcript_status == expected_video_status
        assert video.last_error == error_message

        assert source is not None
        assert source.status == SourceStatus.FAILED