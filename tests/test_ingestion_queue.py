import os
import uuid
from datetime import datetime, timedelta, timezone

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
from src.services.ingestion_queue import run_one_ingestion_cycle
from src.services.ingestion_worker import run_claimed_ingestion_job


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against the configured PostgreSQL database.",
)


@pytest.fixture
def transactional_session_factory():
    """Allow worker commits while rolling all test data back afterward."""
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


def test_queue_cycle_claims_and_executes_one_job(
    transactional_session_factory,
) -> None:
    suffix = uuid.uuid4().hex
    youtube_video_id = suffix[:11]
    now = datetime.now(timezone.utc)

    with transactional_session_factory() as session:
        with session.begin():
            user = User(
                email=f"queue-cycle-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()

            source = create_pending_source(
                session,
                user_id=user.id,
                source_type=SourceType.VIDEO,
                submitted_value=(
                    f"https://www.youtube.com/watch?v={youtube_video_id}"
                ),
            )
            video = get_or_create_video(
                session,
                youtube_video_id=youtube_video_id,
                canonical_url=(
                    f"https://www.youtube.com/watch?v={youtube_video_id}"
                ),
                title="Queue cycle test video",
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
            job.available_at = now - timedelta(seconds=1)

            job_id = job.id
            source_id = source.id
            video_id = video.id

    received_video_ids: list[str] = []

    def fake_ingest(received_video_id: str) -> IngestResult:
        received_video_ids.append(received_video_id)

        return IngestResult(
            video_id=received_video_id,
            status=VideoStatus.OK,
            chunks=2,
            vectors=2,
            title="Indexed by queue cycle",
        )

    def run_fake_job(
        claimed_job_id,
        *,
        claim_attempt: int,
    ) -> IngestResult:
        return run_claimed_ingestion_job(
            claimed_job_id,
            claim_attempt=claim_attempt,
            ingest=fake_ingest,
            session_factory=transactional_session_factory,
        )

    cycle = run_one_ingestion_cycle(
        session_factory=transactional_session_factory,
        run_job=run_fake_job,
        now_fn=lambda: now,
    )

    assert cycle.recovered_job_ids == ()
    assert cycle.job_id == job_id
    assert cycle.claim_attempt == 1
    assert cycle.ingest_result is not None
    assert cycle.ingest_result.status == VideoStatus.OK
    assert received_video_ids == [youtube_video_id]

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

        assert source is not None
        assert source.status == SourceStatus.READY

        assert video is not None
        assert video.transcript_status == TranscriptStatus.READY

        assert source_video is not None
        assert source_video.status == SourceVideoStatus.READY


def test_queue_cycle_exits_cleanly_when_no_job_exists(
    transactional_session_factory,
) -> None:
    called = False

    def fake_run_job(
        job_id,
        *,
        claim_attempt: int,
    ) -> IngestResult:
        nonlocal called
        called = True

        return IngestResult(
            video_id=str(job_id),
            status=VideoStatus.OK,
        )

    cycle = run_one_ingestion_cycle(
        session_factory=transactional_session_factory,
        run_job=fake_run_job,
    )

    assert cycle.recovered_job_ids == ()
    assert cycle.job_id is None
    assert cycle.claim_attempt is None
    assert cycle.ingest_result is None
    assert called is False

def test_queue_cycle_marks_expired_final_attempt_as_failed(
    transactional_session_factory,
) -> None:
    now = datetime(2026, 9, 2, tzinfo=timezone.utc)
    suffix = uuid.uuid4().hex
    youtube_video_id = suffix[:11]

    with transactional_session_factory() as session:
        with session.begin():
            user = User(
                email=f"expired-final-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()

            source = create_pending_source(
                session,
                user_id=user.id,
                source_type=SourceType.VIDEO,
                submitted_value=(
                    f"https://www.youtube.com/watch?v={youtube_video_id}"
                ),
            )
            video = get_or_create_video(
                session,
                youtube_video_id=youtube_video_id,
                canonical_url=(
                    f"https://www.youtube.com/watch?v={youtube_video_id}"
                ),
                title="Expired final-attempt test video",
            )
            source_video = attach_videos(
                session,
                source_id=source.id,
                video_ids=[video.id],
            )[0]
            job = create_ingestion_job(
                session,
                source_id=source.id,
                video_id=video.id,
            )

            # Simulate a worker that had started attempt three, then crashed.
            source.status = SourceStatus.PROCESSING
            source_video.status = SourceVideoStatus.PROCESSING
            video.transcript_status = TranscriptStatus.PROCESSING

            job.status = IngestionJobStatus.RUNNING
            job.attempts = 3
            job.started_at = now - timedelta(minutes=15)
            job.available_at = now - timedelta(minutes=15)
            job.lease_expires_at = now - timedelta(seconds=1)

            job_id = job.id
            source_id = source.id
            video_id = video.id

    called = False

    def fake_run_job(
        claimed_job_id,
        *,
        claim_attempt: int,
    ) -> IngestResult:
        nonlocal called
        called = True

        return IngestResult(
            video_id=str(claimed_job_id),
            status=VideoStatus.OK,
        )

    cycle = run_one_ingestion_cycle(
        session_factory=transactional_session_factory,
        run_job=fake_run_job,
        now_fn=lambda: now,
    )

    assert cycle.recovered_job_ids == (job_id,)
    assert cycle.job_id is None
    assert cycle.claim_attempt is None
    assert cycle.ingest_result is None
    assert called is False

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
        assert job.status == IngestionJobStatus.FAILED
        assert job.error_message == "Worker lease expired before completion."

        assert source_video is not None
        assert source_video.status == SourceVideoStatus.FAILED
        assert source_video.error_message == "Worker lease expired before completion."

        assert video is not None
        assert video.transcript_status == TranscriptStatus.FAILED
        assert video.last_error == "Worker lease expired before completion."

        assert source is not None
        assert source.status == SourceStatus.FAILED