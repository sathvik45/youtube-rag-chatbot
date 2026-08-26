from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from src.db.models.source import Source, SourceStatus
from src.db.models.source_video import SourceVideoStatus
from src.db.models.video import TranscriptStatus
from src.db.repositories.ingestion_jobs import create_ingestion_job
from src.db.repositories.source_videos import update_source_video_status
from src.db.repositories.sources import (
    attach_videos,
    create_pending_source,
    refresh_source_status,
)
from src.db.repositories.videos import get_or_create_video
from src.db.session import SessionLocal
from src.rag.ingest import resolve_videos
from src.youtube.urls import parse_youtube_url


Resolver = Callable[[str], list[str]]
SessionFactory = Callable[[], Session]


@dataclass(frozen=True)
class SourceSubmission:
    source_id: UUID
    video_ids: tuple[UUID, ...]
    job_ids: tuple[UUID, ...]


def _mark_source_failed(
    source_id: UUID,
    *,
    session_factory: SessionFactory,
) -> None:
    session = session_factory()
    try:
        with session.begin():
            source = session.get(Source, source_id)

            if source is None:
                raise LookupError(f"Source {source_id} was not found.")

            source.status = SourceStatus.FAILED
            session.flush()
    finally:
        session.close()


def _unique_video_ids(video_ids: list[str]) -> list[str]:
    """Preserve playlist order while removing duplicate IDs."""
    seen: set[str] = set()

    return [
        video_id
        for video_id in video_ids
        if video_id and not (video_id in seen or seen.add(video_id))
    ]


def submit_source(
    user_id: UUID,
    submitted_value: str,
    *,
    resolve: Resolver = resolve_videos,
    session_factory: SessionFactory = SessionLocal,
) -> SourceSubmission:
    """Create a source, resolve its videos, and queue required ingestion work."""
    parsed = parse_youtube_url(submitted_value)

    # First short transaction: make the user submission durable.
    session = session_factory()
    try:
        with session.begin():
            source = create_pending_source(
                session,
                user_id=user_id,
                submitted_value=submitted_value,
                source_type=parsed.source_type,
                youtube_playlist_id=parsed.playlist_id,
            )
            source_id = source.id
    finally:
        session.close()

    # This may contact Supadata for a playlist. No database transaction is open.
    try:
        youtube_video_ids = _unique_video_ids(resolve(submitted_value))
    except Exception:
        _mark_source_failed(source_id, session_factory=session_factory)
        raise

    if not youtube_video_ids:
        _mark_source_failed(source_id, session_factory=session_factory)
        raise ValueError("The submitted source did not resolve to any videos.")

    # Second short transaction: create/reuse rows, links, and jobs.
    session = session_factory()
    try:
        with session.begin():
            videos = [
                get_or_create_video(
                    session,
                    youtube_video_id=youtube_video_id,
                    canonical_url=(
                        f"https://www.youtube.com/watch?v={youtube_video_id}"
                    ),
                    title=None,
                )
                for youtube_video_id in youtube_video_ids
            ]

            source_videos = attach_videos(
                session,
                source_id=source_id,
                video_ids=[video.id for video in videos],
            )

            job_ids: list[UUID] = []

            for video, source_video in zip(videos, source_videos):
                if video.transcript_status == TranscriptStatus.READY:
                    update_source_video_status(
                        session,
                        source_id=source_id,
                        video_id=video.id,
                        status=SourceVideoStatus.READY,
                    )
                elif video.transcript_status == TranscriptStatus.NO_TRANSCRIPT:
                    update_source_video_status(
                        session,
                        source_id=source_id,
                        video_id=video.id,
                        status=SourceVideoStatus.NO_TRANSCRIPT,
                        error_message=video.last_error,
                    )
                else:
                    job = create_ingestion_job(
                        session,
                        source_id=source_id,
                        video_id=video.id,
                    )
                    job_ids.append(job.id)

            refresh_source_status(
                session,
                source_id=source_id,
            )

            return SourceSubmission(
                source_id=source_id,
                video_ids=tuple(video.id for video in videos),
                job_ids=tuple(job_ids),
            )
    finally:
        session.close()