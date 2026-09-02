from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models.ingestion_job import IngestionJob, IngestionJobStatus

DEFAULT_INGESTION_JOB_LEASE = timedelta(minutes=15)
DEFAULT_INGESTION_RETRY_DELAY = timedelta(seconds=30)
MAX_INGESTION_JOB_ATTEMPTS = 3
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

def claim_next_ingestion_job(
    session: Session,
    *,
    now: datetime,
    lease_duration: timedelta,
) -> IngestionJob | None:
    """Atomically claim the oldest eligible ingestion job."""

    if lease_duration <= timedelta():
        raise ValueError("lease_duration must be positive.")

    statement = (
        select(IngestionJob)
        .where(
            IngestionJob.status.in_(
                [
                    IngestionJobStatus.QUEUED,
                    IngestionJobStatus.RETRYING,
                ]
            ),
            IngestionJob.available_at <= now,
        )
        .order_by(
            IngestionJob.available_at,
            IngestionJob.created_at,
            IngestionJob.id,
        )
        .with_for_update(skip_locked=True)
        .limit(1)
    )

    job = session.scalar(statement)

    if job is None:
        return None

    job.status = IngestionJobStatus.RUNNING
    job.attempts += 1
    job.started_at = now
    job.finished_at = None
    job.lease_expires_at = now + lease_duration
    job.error_message = None

    session.flush()

    return job

def mark_job_running(
    session: Session,
    *,
    job_id: UUID,
    lease_duration: timedelta = DEFAULT_INGESTION_JOB_LEASE,
) -> IngestionJob:
    """Mark one eligible job as running and give it a worker lease."""

    if lease_duration <= timedelta():
        raise ValueError("lease_duration must be positive.")

    statement = (
        select(IngestionJob)
        .where(IngestionJob.id == job_id)
        .with_for_update()
    )
    job = session.scalar(statement)

    if job is None:
        raise LookupError(f"Ingestion job {job_id} was not found.")

    if job.status not in {
        IngestionJobStatus.QUEUED,
        IngestionJobStatus.RETRYING,
    }:
        raise ValueError(
            f"Only queued or retrying jobs can start; "
            f"job {job_id} is {job.status.value}."
        )

    started_at = datetime.now(timezone.utc)

    if job.available_at > started_at:
        raise ValueError(
            f"Ingestion job {job_id} is not available to run yet."
        )

    job.status = IngestionJobStatus.RUNNING
    job.attempts += 1
    job.started_at = started_at
    job.finished_at = None
    job.lease_expires_at = started_at + lease_duration
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
    job.lease_expires_at = None
    session.flush()

    return job

def mark_job_retrying(
    session: Session,
    *,
    job_id: UUID,
    error_message: str,
    retry_delay: timedelta = DEFAULT_INGESTION_RETRY_DELAY,
) -> IngestionJob:
    """Record a retryable failure and schedule its next attempt."""

    if retry_delay <= timedelta():
        raise ValueError("retry_delay must be positive.")

    job = session.get(IngestionJob, job_id)

    if job is None:
        raise LookupError(f"Ingestion job {job_id} was not found.")

    if job.status != IngestionJobStatus.RUNNING:
        raise ValueError(
            f"Only running jobs can retry; job {job_id} is {job.status.value}."
        )

    finished_at = datetime.now(timezone.utc)

    job.status = IngestionJobStatus.RETRYING
    job.error_message = error_message
    job.finished_at = finished_at
    job.available_at = finished_at + retry_delay
    job.lease_expires_at = None

    session.flush()

    return job

def mark_job_failed(
    session: Session,
    *,
    job_id: UUID,
    error_message: str,
) -> IngestionJob:
    """Record a terminal failure for a running ingestion job."""
    job = session.get(IngestionJob, job_id)

    if job is None:
        raise LookupError(f"Ingestion job {job_id} was not found.")

    if job.status != IngestionJobStatus.RUNNING:
        raise ValueError(
            f"Only running jobs can fail; job {job_id} is {job.status.value}."
        )

    job.status = IngestionJobStatus.FAILED
    job.error_message = error_message
    job.finished_at = datetime.now(timezone.utc)
    job.lease_expires_at = None

    session.flush()

    return job

def requeue_expired_ingestion_jobs(
    session: Session,
    *,
    now: datetime,
    retry_delay: timedelta = DEFAULT_INGESTION_RETRY_DELAY,
) -> list[IngestionJob]:
    """Move abandoned running jobs back to retrying."""

    if retry_delay <= timedelta():
        raise ValueError("retry_delay must be positive.")

    statement = (
        select(IngestionJob)
        .where(
            IngestionJob.status == IngestionJobStatus.RUNNING,
            IngestionJob.lease_expires_at.is_not(None),
            IngestionJob.lease_expires_at <= now,
        )
        .order_by(
            IngestionJob.lease_expires_at,
            IngestionJob.id,
        )
        .with_for_update(skip_locked=True)
    )

    jobs = list(session.scalars(statement))

    for job in jobs:
        job.status = IngestionJobStatus.RETRYING
        job.finished_at = now
        job.available_at = now + retry_delay
        job.lease_expires_at = None
        job.error_message = "Worker lease expired before completion."

    session.flush()

    return jobs

def list_runnable_jobs(
    session: Session,
    *,
    source_id: UUID,
    now: datetime | None = None,
) -> list[IngestionJob]:
    """Return jobs that are eligible to run for one source."""

    runnable_at = now or datetime.now(timezone.utc)

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
            IngestionJob.available_at <= runnable_at,
        )
        .order_by(
            IngestionJob.available_at,
            IngestionJob.created_at,
            IngestionJob.id,
        )
    )

    return list(session.scalars(statement))