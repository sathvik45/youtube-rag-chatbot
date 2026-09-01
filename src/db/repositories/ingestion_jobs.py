from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from datetime import datetime, timezone

from src.db.models.ingestion_job import IngestionJob, IngestionJobStatus


def create_ingestion_job(
    session: Session,
    *,
    source_id: UUID,
    video_id: UUID,
) -> IngestionJob:
    """Queue ingestion work for one source/video pair."""
    job = IngestionJob(
        source_id=source_id,
        video_id=video_id,
    )
    session.add(job)
    session.flush()

    return job

def mark_job_running(
    session: Session,
    *,
    job_id: UUID,
) -> IngestionJob:
    """Claim a queued job before starting its ingestion work."""
    job = session.get(IngestionJob, job_id)

    if job is None:
        raise LookupError(f"Ingestion job {job_id} was not found.")

    if job.status not in {IngestionJobStatus.QUEUED, IngestionJobStatus.RETRYING}:
        raise ValueError(
            f"Only queued or retrying jobs can start; "f"job {job_id} is {job.status.value}."
        )

    job.status = IngestionJobStatus.RUNNING
    job.attempts += 1
    job.started_at = datetime.now(timezone.utc)
    job.error_message = None

    session.flush()

    return job

def mark_job_succeeded(
    session: Session,
    *,
    job_id: UUID,
) -> IngestionJob:
    """Mark a running ingestion job as completed successfully."""
    job = session.get(IngestionJob, job_id)

    if job is None:
        raise LookupError(f"Ingestion job {job_id} was not found.")

    if job.status != IngestionJobStatus.RUNNING:
        raise ValueError(
            f"Only running jobs can succeed; job {job_id} is {job.status.value}."
        )

    job.status = IngestionJobStatus.SUCCEEDED
    job.finished_at = datetime.now(timezone.utc)
    job.error_message = None

    session.flush()

    return job

def mark_job_retrying(
    session: Session,
    *,
    job_id: UUID,
    error_message: str,
) -> IngestionJob:
    """Record a retryable ingestion failure."""
    job = session.get(IngestionJob, job_id)

    if job is None:
        raise LookupError(f"Ingestion job {job_id} was not found.")

    if job.status != IngestionJobStatus.RUNNING:
        raise ValueError(
            f"Only running jobs can retry; job {job_id} is {job.status.value}."
        )

    job.status = IngestionJobStatus.RETRYING
    job.error_message = error_message
    job.finished_at = datetime.now(timezone.utc)

    session.flush()

    return job

def list_runnable_jobs(
    session: Session,
    *,
    source_id: UUID,
) -> list[IngestionJob]:
    """Return jobs that may be processed for one source."""
    statement = (
        select(IngestionJob)
        .where(
            IngestionJob.source_id == source_id,
            IngestionJob.status.in_(
                [
                    IngestionJobStatus.QUEUED,
                    IngestionJobStatus.RETRYING,
                ]
            ),
        )
        .order_by(IngestionJob.created_at, IngestionJob.id)
    )

    return list(session.scalars(statement))