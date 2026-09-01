from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models.source import Source, SourceStatus, SourceType
from src.db.models.source_video import SourceVideo, SourceVideoStatus
from src.db.models.video import Video

from dataclasses import dataclass
from src.db.models.ingestion_job import IngestionJob, IngestionJobStatus


@dataclass(frozen=True)
class SourceProgress:
    source_id: UUID
    source_status: SourceStatus
    video_counts: dict[str, int]
    job_counts: dict[str, int]

    @property
    def total_videos(self) -> int:
        return sum(self.video_counts.values())

    @property
    def completed_videos(self) -> int:
        return (
            self.video_counts[SourceVideoStatus.READY.value]
            + self.video_counts[SourceVideoStatus.NO_TRANSCRIPT.value]
            + self.video_counts[SourceVideoStatus.FAILED.value]
        )

def create_source(
    db: Session,
    user_id: UUID,
    source_type: SourceType,
    submitted_value: str,
    youtube_playlist_id: str | None = None,
) -> Source:
    source = Source(
        user_id=user_id,
        source_type=source_type,
        submitted_value=submitted_value,
        youtube_playlist_id=youtube_playlist_id,
    )

    db.add(source)
    db.commit()
    db.refresh(source)

    return source


def attach_videos(
    db: Session,
    source_id: UUID,
    video_ids: list[UUID],
) -> list[SourceVideo]:
    source_videos = [
        SourceVideo(
            source_id=source_id,
            video_id=video_id,
            position=position,
        )
        for position, video_id in enumerate(video_ids)
    ]

    db.add_all(source_videos)
    db.flush()
    return source_videos


def get_source(
    db: Session,
    source_id: UUID,
) -> Source | None:
    statement = select(Source).where(Source.id == source_id)

    return db.scalar(statement)


def get_source_progress(
    session: Session,
    *,
    source_id: UUID,
) -> SourceProgress:
    """Return status counts for one source and its ingestion jobs."""
    source = session.get(Source, source_id)

    if source is None:
        raise LookupError(f"Source {source_id} was not found.")

    video_counts = {
        status.value: 0
        for status in SourceVideoStatus
    }
    job_counts = {
        status.value: 0
        for status in IngestionJobStatus
    }

    source_video_rows = session.execute(
        select(
            SourceVideo.status,
            func.count(SourceVideo.id),
        )
        .where(SourceVideo.source_id == source_id)
        .group_by(SourceVideo.status)
    ).all()

    for status, count in source_video_rows:
        video_counts[status.value] = count

    job_rows = session.execute(
        select(
            IngestionJob.status,
            func.count(IngestionJob.id),
        )
        .where(IngestionJob.source_id == source_id)
        .group_by(IngestionJob.status)
    ).all()

    for status, count in job_rows:
        job_counts[status.value] = count

    return SourceProgress(
        source_id=source.id,
        source_status=source.status,
        video_counts=video_counts,
        job_counts=job_counts,
    )


def create_pending_source(
    session: Session,
    *,
    user_id: UUID,
    submitted_value: str,
    source_type: SourceType,
    youtube_playlist_id: str | None = None,
) -> Source:
    """Save a user-submitted source in its initial pending state."""
    submitted_value = submitted_value.strip()

    if not submitted_value:
        raise ValueError("A YouTube URL is required.")

    source = Source(
        user_id=user_id,
        source_type=source_type,
        submitted_value=submitted_value,
        youtube_playlist_id=youtube_playlist_id,
    )
    session.add(source)
    session.flush()

    return source

def refresh_source_status(
    session: Session,
    *,
    source_id: UUID,
) -> Source:
    """Recalculate one source's status from its linked video states."""
    source = session.get(Source, source_id)

    if source is None:
        raise LookupError(f"Source {source_id} was not found.")

    video_statuses = session.scalars(
        select(SourceVideo.status).where(SourceVideo.source_id == source_id)
    ).all()

    if not video_statuses:
        source.status = SourceStatus.PENDING
    elif any(
        status in {SourceVideoStatus.PENDING, SourceVideoStatus.PROCESSING}
        for status in video_statuses
    ):
        source.status = SourceStatus.PROCESSING
    elif all(status == SourceVideoStatus.READY for status in video_statuses):
        source.status = SourceStatus.READY
    elif any(status == SourceVideoStatus.READY for status in video_statuses):
        source.status = SourceStatus.PARTIAL
    else:
        source.status = SourceStatus.FAILED

    session.flush()

    return source