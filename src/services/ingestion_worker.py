from collections.abc import Callable
from uuid import UUID

from sqlalchemy.orm import Session

from src.db.models.ingestion_job import IngestionJob
from src.db.models.source_video import SourceVideoStatus
from src.db.models.video import TranscriptStatus, Video
from src.db.repositories.ingestion_jobs import (
    mark_job_retrying,
    mark_job_running,
    mark_job_succeeded,
)
from src.db.repositories.source_videos import update_source_video_status
from src.db.repositories.sources import refresh_source_status
from src.db.repositories.videos import mark_video_ready, update_video_status
from src.db.session import SessionLocal
from src.rag.ingest import IngestResult, VideoStatus, ingest_video


IngestFunction = Callable[[str], IngestResult]
SessionFactory = Callable[[], Session]


def _get_job_and_video(
    session: Session,
    job_id: UUID,
) -> tuple[IngestionJob, Video]:
    job = session.get(IngestionJob, job_id)

    if job is None:
        raise LookupError(f"Ingestion job {job_id} was not found.")

    if job.video_id is None:
        raise ValueError(f"Ingestion job {job_id} has no video.")

    video = session.get(Video, job.video_id)

    if video is None:
        raise LookupError(f"Video {job.video_id} was not found.")

    return job, video


def _record_started(session: Session, job_id: UUID) -> str:
    """Mark database records as processing and return the YouTube video ID."""
    job, video = _get_job_and_video(session, job_id)

    mark_job_running(session, job_id=job.id)

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

    return video.youtube_video_id


def _record_result(
    session: Session,
    job_id: UUID,
    result: IngestResult,
) -> None:
    """Save one completed RAG ingestion result."""
    job, video = _get_job_and_video(session, job_id)
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

        # The job completed correctly, even though captions do not exist.
        mark_job_succeeded(session, job_id=job.id)

    else:
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

        mark_job_retrying(
            session,
            job_id=job.id,
            error_message=error_message,
        )

    refresh_source_status(
        session,
        source_id=job.source_id,
    )


def run_ingestion_job(
    job_id: UUID,
    *,
    ingest: IngestFunction = ingest_video,
    session_factory: SessionFactory = SessionLocal,
) -> IngestResult:
    """Run one queued ingestion job and persist its outcome."""
    session = session_factory()
    try:
        with session.begin():
            youtube_video_id = _record_started(session, job_id)
    finally:
        session.close()

    # This may contact Supadata, embed chunks, and update Pinecone.
    # It deliberately runs with no database transaction open.
    result = ingest(youtube_video_id)

    session = session_factory()
    try:
        with session.begin():
            _record_result(session, job_id, result)
    finally:
        session.close()

    return result