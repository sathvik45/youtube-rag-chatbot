from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from src.db.repositories.ingestion_jobs import (
    DEFAULT_INGESTION_JOB_LEASE,
    DEFAULT_INGESTION_RETRY_DELAY,
    claim_next_ingestion_job,
    requeue_expired_ingestion_jobs,
)
from src.db.session import SessionLocal
from src.rag.ingest import IngestResult
from src.services.ingestion_worker import run_claimed_ingestion_job


SessionFactory = Callable[[], Session]
Clock = Callable[[], datetime]


class ClaimedJobRunner(Protocol):
    def __call__(
        self,
        job_id: UUID,
        *,
        claim_attempt: int,
    ) -> IngestResult: ...


@dataclass(frozen=True)
class WorkerCycleResult:
    recovered_job_ids: tuple[UUID, ...]
    job_id: UUID | None
    claim_attempt: int | None
    ingest_result: IngestResult | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def run_one_ingestion_cycle(
    *,
    session_factory: SessionFactory = SessionLocal,
    run_job: ClaimedJobRunner = run_claimed_ingestion_job,
    now_fn: Clock = utc_now,
    lease_duration: timedelta = DEFAULT_INGESTION_JOB_LEASE,
    retry_delay: timedelta = DEFAULT_INGESTION_RETRY_DELAY,
) -> WorkerCycleResult:
    """Recover stale work, claim at most one job, and run that claim."""
    now = now_fn()

    session = session_factory()
    try:
        with session.begin():
            recovered_jobs = requeue_expired_ingestion_jobs(
                session,
                now=now,
                retry_delay=retry_delay,
            )
            recovered_job_ids = tuple(job.id for job in recovered_jobs)

            claimed_job = claim_next_ingestion_job(
                session,
                now=now,
                lease_duration=lease_duration,
            )

            if claimed_job is None:
                job_id = None
                claim_attempt = None
            else:
                job_id = claimed_job.id
                claim_attempt = claimed_job.attempts
    finally:
        session.close()

    if job_id is None or claim_attempt is None:
        return WorkerCycleResult(
            recovered_job_ids=recovered_job_ids,
            job_id=None,
            claim_attempt=None,
            ingest_result=None,
        )

    ingest_result = run_job(
        job_id,
        claim_attempt=claim_attempt,
    )

    return WorkerCycleResult(
        recovered_job_ids=recovered_job_ids,
        job_id=job_id,
        claim_attempt=claim_attempt,
        ingest_result=ingest_result,
    )