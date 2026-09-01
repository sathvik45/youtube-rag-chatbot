from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from src.db.repositories.ingestion_jobs import list_runnable_jobs
from src.db.session import SessionLocal
from src.rag.ingest import IngestResult
from src.services.ingestion_worker import run_ingestion_job


JobRunner = Callable[[UUID], IngestResult]
SessionFactory = Callable[[], Session]


def run_runnable_jobs_for_source(
    source_id: UUID,
    *,
    run_job: JobRunner = run_ingestion_job,
    session_factory: SessionFactory = SessionLocal,
) -> list[IngestResult]:
    """Run every queued or retrying ingestion job for one source."""
    session = session_factory()
    try:
        job_ids = [
            job.id
            for job in list_runnable_jobs(
                session,
                source_id=source_id,
            )
        ]
    finally:
        session.close()

    return [run_job(job_id) for job_id in job_ids]