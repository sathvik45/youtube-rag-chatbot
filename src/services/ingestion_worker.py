from collections.abc import Callable
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models.ingestion_job import IngestionJob, IngestionJobStatus
from src.db.models.source_video import SourceVideoStatus
from src.db.models.video import TranscriptStatus, Video
from src.db.repositories.ingestion_jobs import (
    mark_job_retrying,
    mark_job_running,
    mark_job_succeeded,
    MAX_INGESTION_JOB_ATTEMPTS,
    mark_job_failed,
)
from src.db.repositories.source_videos import update_source_video_status
from src.db.repositories.sources import refresh_source_status
from src.db.repositories.videos import mark_video_ready, update_video_status
from src.db.session import SessionLocal
from src.rag.ingest import IngestResult, VideoStatus, ingest_video


IngestFunction = Callable[[str], IngestResult]
SessionFactory = Callable[[], Session]


class IngestionJobClaimLostError(RuntimeError):
    """Raised when another worker owns the job attempt now."""


def _get_job_and_video(
    session: Session,
    job_id: UUID,
    *,
    lock_job: bool = False,
) -> tuple[IngestionJob, Video]:
    if lock_job:
        statement = (
            select(IngestionJob)
            .where(IngestionJob.id == job_id)
            .with_for_update()
        )
        job = session.scalar(statement)
    else:
        job = session.get(IngestionJob, job_id)

    if job is None:
        raise LookupError(f"Ingestion job {job_id} was not found.")

    if job.video_id is None:
        raise ValueError(f"Ingestion job {job_id} has no video.")

    video = session.get(Video, job.video_id)

    if video is None:
        raise LookupError(f"Video {job.video_id} was not found.")

    return job, video


def _assert_current_claim(
    job: IngestionJob,
    *,
    job_id: UUID,
    claim_attempt: int,
    require_active_lease: bool,
) -> None:
    if job.status != IngestionJobStatus.RUNNING:
        raise IngestionJobClaimLostError(
            f"Ingestion job {job_id} is no longer running."
        )

    if job.attempts != claim_attempt:
        raise IngestionJobClaimLostError(
            f"Ingestion job {job_id} now belongs to attempt {job.attempts}, "
            f"not attempt {claim_attempt}."
        )

    if not require_active_lease:
        return

    if job.lease_expires_at is None:
        raise IngestionJobClaimLostError(
            f"Ingestion job {job_id} has no worker lease."
        )

    if job.lease_expires_at <= datetime.now(timezone.utc):
        raise IngestionJobClaimLostError(
            f"Ingestion job {job_id} has an expired worker lease."
        )


def _record_processing_state(
    session: Session,
    *,
    job: IngestionJob,
    video: Video,
) -> None:
    update_source_video_status(
        session,
        source_id=job.source_id,
        video_id=video.id,
        status=SourceVideoStatus.PROCESSING,
    )

    update_video_status(
        session,
        video,
        TranscriptStatus.PROCESSING,
    )

    refresh_source_status(
        session,
        source_id=job.source_id,
    )


def _record_started(session: Session, job_id: UUID) -> tuple[str, int]:
    """Claim one queued job and record that its work has started."""
    job, video = _get_job_and_video(session, job_id)

    claimed_job = mark_job_running(session, job_id=job.id)

    _record_processing_state(
        session,
        job=claimed_job,
        video=video,
    )

    return video.youtube_video_id, claimed_job.attempts


def _record_claimed_started(
    session: Session,
    job_id: UUID,
    *,
    claim_attempt: int,
) -> str:
    """Record processing state for an already atomically claimed job."""
    job, video = _get_job_and_video(
        session,
        job_id,
        lock_job=True,
    )

    _assert_current_claim(
        job,
        job_id=job_id,
        claim_attempt=claim_attempt,
        require_active_lease=True,
    )

    _record_processing_state(
        session,
        job=job,
        video=video,
    )

    return video.youtube_video_id


def _record_result(
    session: Session,
    job_id: UUID,
    *,
    claim_attempt: int,
    result: IngestResult,
) -> None:
    """Save a result only when this worker still owns the job attempt."""
    job, video = _get_job_and_video(
        session,
        job_id,
        lock_job=True,
    )

    _assert_current_claim(
        job,
        job_id=job_id,
        claim_attempt=claim_attempt,
        require_active_lease=False,
    )

    error_message = result.message or result.error or "Ingestion failed."

    if result.status is VideoStatus.OK:
        mark_video_ready(
            session,
            video,
            chunk_count=result.chunks,
            vector_count=result.vectors,
            title=result.title,
        )

        update_source_video_status(
            session,
            source_id=job.source_id,
            video_id=video.id,
            status=SourceVideoStatus.READY,
        )

        mark_job_succeeded(session, job_id=job.id)

    elif result.status is VideoStatus.NO_TRANSCRIPT:
        update_video_status(
            session,
            video,
            TranscriptStatus.NO_TRANSCRIPT,
            last_error=error_message,
        )

        update_source_video_status(
            session,
            source_id=job.source_id,
            video_id=video.id,
            status=SourceVideoStatus.NO_TRANSCRIPT,
            error_message=error_message,
        )

        mark_job_succeeded(session, job_id=job.id)

    else:
        if job.attempts >= MAX_INGESTION_JOB_ATTEMPTS:
            mark_job_failed(
                session,
                job_id=job.id,
                error_message=error_message,
            )

            update_video_status(
                session,
                video,
                TranscriptStatus.FAILED,
                last_error=error_message,
            )

            update_source_video_status(
                session,
                source_id=job.source_id,
                video_id=video.id,
                status=SourceVideoStatus.FAILED,
                error_message=error_message,
            )
        else:
            mark_job_retrying(
                session,
                job_id=job.id,
                error_message=error_message,
            )

            update_video_status(
                session,
                video,
                TranscriptStatus.PROCESSING,
                last_error=error_message,
            )

            update_source_video_status(
                session,
                source_id=job.source_id,
                video_id=video.id,
                status=SourceVideoStatus.PROCESSING,
                error_message=error_message,
            )

    refresh_source_status(
        session,
        source_id=job.source_id,
    )


def _ingest_and_record(
    job_id: UUID,
    *,
    claim_attempt: int,
    youtube_video_id: str,
    ingest: IngestFunction,
    session_factory: SessionFactory,
) -> IngestResult:
    """Run external work, then persist the outcome in a new transaction."""
    try:
        result = ingest(youtube_video_id)
    except Exception as error:
        result = IngestResult(
            video_id=youtube_video_id,
            status=VideoStatus.ERROR,
            error=type(error).__name__,
            message="Unexpected ingestion error.",
        )

    session = session_factory()
    try:
        with session.begin():
            _record_result(
                session,
                job_id,
                claim_attempt=claim_attempt,
                result=result,
            )
    finally:
        session.close()

    return result


def run_ingestion_job(
    job_id: UUID,
    *,
    ingest: IngestFunction = ingest_video,
    session_factory: SessionFactory = SessionLocal,
) -> IngestResult:
    """Claim and run one queued or retrying ingestion job."""
    session = session_factory()
    try:
        with session.begin():
            youtube_video_id, claim_attempt = _record_started(session, job_id)
    finally:
        session.close()

    return _ingest_and_record(
        job_id,
        claim_attempt=claim_attempt,
        youtube_video_id=youtube_video_id,
        ingest=ingest,
        session_factory=session_factory,
    )


def run_claimed_ingestion_job(
    job_id: UUID,
    *,
    claim_attempt: int,
    ingest: IngestFunction = ingest_video,
    session_factory: SessionFactory = SessionLocal,
) -> IngestResult:
    """Run a job already atomically claimed by a queue worker."""
    session = session_factory()
    try:
        with session.begin():
            youtube_video_id = _record_claimed_started(
                session,
                job_id,
                claim_attempt=claim_attempt,
            )
    finally:
        session.close()

    return _ingest_and_record(
        job_id,
        claim_attempt=claim_attempt,
        youtube_video_id=youtube_video_id,
        ingest=ingest,
        session_factory=session_factory,
    )