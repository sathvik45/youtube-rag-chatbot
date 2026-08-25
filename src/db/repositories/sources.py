from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.models.source import Source, SourceStatus, SourceType
from src.models.source_video import SourceVideo
from src.models.video import Video


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
        for position, video_id in enumerate(video_ids, start=1)
    ]

    db.add_all(source_videos)
    db.commit()

    for source_video in source_videos:
        db.refresh(source_video)

    return source_videos


def get_source(
    db: Session,
    source_id: UUID,
) -> Source | None:
    statement = select(Source).where(Source.id == source_id)

    return db.scalar(statement)


def get_ingestion_progress(
    db: Session,
    source_id: UUID,
) -> dict:
    statement = (
        select(
            SourceVideo.status,
            func.count(SourceVideo.id),
        )
        .where(SourceVideo.source_id == source_id)
        .group_by(SourceVideo.status)
    )

    rows = db.execute(statement).all()

    progress = {
        "pending": 0,
        "processing": 0,
        "ready": 0,
        "failed": 0,
    }

    for status, count in rows:
        progress[status] = count

    total = sum(progress.values())

    completed = (
        progress["ready"]
        + progress["failed"]
    )

    return {
        "total": total,
        "pending": progress["pending"],
        "processing": progress["processing"],
        "ready": progress["ready"],
        "failed": progress["failed"],
        "completed": completed,
        "progress_percent": (
            (completed / total) * 100
            if total > 0
            else 0
        ),
    }