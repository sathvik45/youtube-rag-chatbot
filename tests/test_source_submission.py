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
from src.db.repositories.videos import get_or_create_video, mark_video_ready

from src.rag.ingest import IngestResult, VideoStatus
from src.services.ingestion_dispatcher import run_runnable_jobs_for_source

from src.services.ingestion_worker import run_ingestion_job

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

def test_submit_source_reuses_ready_video_without_creating_job(
    transactional_session_factory,
) -> None:
    suffix = uuid.uuid4().hex
    youtube_video_id = suffix[:11]
    submitted_url = f"https://www.youtube.com/watch?v={youtube_video_id}"

    with transactional_session_factory() as session:
        with session.begin():
            user = User(
                email=f"reuse-test-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()

            existing_video = get_or_create_video(
                session,
                youtube_video_id=youtube_video_id,
                canonical_url=submitted_url,
                title="Previously indexed video",
            )
            mark_video_ready(
                session,
                existing_video,
                chunk_count=5,
                vector_count=5,
            )

            user_id = user.id
            existing_video_id = existing_video.id

    def fake_resolve(received_url: str) -> list[str]:
        assert received_url == submitted_url
        return [youtube_video_id]

    submission = submit_source(
        user_id,
        submitted_url,
        resolve=fake_resolve,
        session_factory=transactional_session_factory,
    )

    assert submission.video_ids == (existing_video_id,)
    assert submission.job_ids == ()

    with transactional_session_factory() as session:
        source = session.get(Source, submission.source_id)
        videos = session.scalars(
            select(Video).where(Video.youtube_video_id == youtube_video_id)
        ).all()
        source_video = session.scalar(
            select(SourceVideo).where(
                SourceVideo.source_id == submission.source_id,
                SourceVideo.video_id == existing_video_id,
            )
        )
        jobs = session.scalars(
            select(IngestionJob).where(
                IngestionJob.source_id == submission.source_id
            )
        ).all()

        assert source is not None
        assert source.source_type == SourceType.VIDEO
        assert source.status == SourceStatus.READY

        assert [video.id for video in videos] == [existing_video_id]
        assert videos[0].transcript_status == TranscriptStatus.READY

        assert source_video is not None
        assert source_video.status == SourceVideoStatus.READY

        assert jobs == []

def test_submit_source_marks_source_failed_when_resolution_fails(
    transactional_session_factory,
) -> None:
    suffix = uuid.uuid4().hex
    youtube_video_id = suffix[:11]
    submitted_url = f"https://www.youtube.com/watch?v={youtube_video_id}"

    with transactional_session_factory() as session:
        with session.begin():
            user = User(
                email=f"resolution-failure-test-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()

            user_id = user.id

    def fake_resolve(received_url: str) -> list[str]:
        assert received_url == submitted_url
        raise RuntimeError("YouTube resolver is unavailable.")

    with pytest.raises(RuntimeError, match="resolver is unavailable"):
        submit_source(
            user_id,
            submitted_url,
            resolve=fake_resolve,
            session_factory=transactional_session_factory,
        )

    with transactional_session_factory() as session:
        source = session.scalar(
            select(Source).where(
                Source.user_id == user_id,
                Source.submitted_value == submitted_url,
            )
        )
        source_videos = session.scalars(
            select(SourceVideo).where(SourceVideo.source_id == source.id)
        ).all()
        jobs = session.scalars(
            select(IngestionJob).where(IngestionJob.source_id == source.id)
        ).all()

        assert source is not None

        source_videos = session.scalars(
            select(SourceVideo).where(SourceVideo.source_id == source.id)
        ).all()
        jobs = session.scalars(
            select(IngestionJob).where(IngestionJob.source_id == source.id)
        ).all()
        assert source.source_type == SourceType.VIDEO
        assert source.status == SourceStatus.FAILED
        assert source_videos == []
        assert jobs == []

def test_dispatcher_runs_only_queued_and_retrying_jobs(
    transactional_session_factory,
) -> None:
    suffix = uuid.uuid4().hex
    submitted_url = "https://www.youtube.com/playlist?list=PL123456789"
    resolved_video_ids = [
        "dQw4w9WgXcQ",
        "9bZkp7q19f0",
        "M7lc1UVf-VE",
    ]

    with transactional_session_factory() as session:
        with session.begin():
            user = User(
                email=f"dispatcher-test-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()

            user_id = user.id

    submission = submit_source(
        user_id,
        submitted_url,
        resolve=lambda _: resolved_video_ids,
        session_factory=transactional_session_factory,
    )

    with transactional_session_factory() as session:
        with session.begin():
            jobs = session.scalars(
                select(IngestionJob)
                .where(IngestionJob.source_id == submission.source_id)
                .order_by(IngestionJob.created_at, IngestionJob.id)
            ).all()

            jobs[0].status = IngestionJobStatus.SUCCEEDED
            jobs[1].status = IngestionJobStatus.RETRYING
            session.flush()

            expected_job_ids = [jobs[1].id, jobs[2].id]

    called_job_ids: list[uuid.UUID] = []

    def fake_run_job(job_id: uuid.UUID) -> IngestResult:
        called_job_ids.append(job_id)

        return IngestResult(
            video_id=str(job_id),
            status=VideoStatus.OK,
        )

    results = run_runnable_jobs_for_source(
        submission.source_id,
        run_job=fake_run_job,
        session_factory=transactional_session_factory,
    )

    assert called_job_ids == expected_job_ids
    assert len(results) == 2
    assert all(result.status == VideoStatus.OK for result in results) 

def test_source_submission_dispatches_and_completes_ingestion(
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
                email=f"workflow-test-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(user)
            session.flush()

            user_id = user.id

    submission = submit_source(
        user_id,
        submitted_url,
        resolve=lambda _: resolved_video_ids,
        session_factory=transactional_session_factory,
    )

    received_video_ids: list[str] = []

    def fake_ingest(youtube_video_id: str) -> IngestResult:
        received_video_ids.append(youtube_video_id)

        return IngestResult(
            video_id=youtube_video_id,
            status=VideoStatus.OK,
            chunks=4,
            vectors=4,
            title=f"Indexed {youtube_video_id}",
        )

    def run_fake_worker(job_id: uuid.UUID) -> IngestResult:
        return run_ingestion_job(
            job_id,
            ingest=fake_ingest,
            session_factory=transactional_session_factory,
        )

    results = run_runnable_jobs_for_source(
        submission.source_id,
        run_job=run_fake_worker,
        session_factory=transactional_session_factory,
    )

    assert len(results) == 2
    assert all(result.status == VideoStatus.OK for result in results)
    assert set(received_video_ids) == set(resolved_video_ids)

    with transactional_session_factory() as session:
        source = session.get(Source, submission.source_id)
        videos = session.scalars(
            select(Video).where(Video.id.in_(submission.video_ids))
        ).all()
        source_videos = session.scalars(
            select(SourceVideo).where(
                SourceVideo.source_id == submission.source_id
            )
        ).all()
        jobs = session.scalars(
            select(IngestionJob).where(
                IngestionJob.source_id == submission.source_id
            )
        ).all()

        assert source is not None
        assert source.status == SourceStatus.READY

        assert len(videos) == 2
        assert all(
            video.transcript_status == TranscriptStatus.READY
            for video in videos
        )
        assert all(video.chunk_count == 4 for video in videos)
        assert all(video.vector_count == 4 for video in videos)

        assert len(source_videos) == 2
        assert all(
            source_video.status == SourceVideoStatus.READY
            for source_video in source_videos
        )

        assert len(jobs) == 2
        assert all(
            job.status == IngestionJobStatus.SUCCEEDED
            for job in jobs
        )
        assert all(job.attempts == 1 for job in jobs)