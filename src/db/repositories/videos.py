from sqlalchemy.orm import Session
from sqlalchemy import select

from src.db.models.video import Video, TranscriptStatus

from datetime import datetime, timezone

def get_video_by_id(db : Session, youtube_video_id : str) -> Video | None:
    statement = select(Video).where(Video.youtube_video_id == youtube_video_id)
    return db.scalar(statement)

def create_video(db : Session, youtube_video_id : str, canonical_url : str, title : str, transcript_language : str | None = None) -> Video:
    video = Video(youtube_video_id=youtube_video_id,canonical_url=canonical_url,title=title,transcript_language=transcript_language)
    db.add(video)
    db.commit()
    db.refresh(video)
    return video

def get_or_create_video(db : Session,youtube_video_id : str, canonical_url : str, title : str, transcript_language : str | None = None)-> Video:
    video = get_video_by_id(db=db,youtube_video_id=youtube_video_id)
    if video is not None:
        return video
    return create_video(db=db, youtube_video_id=youtube_video_id, canonical_url=canonical_url, title=title,transcript_language=transcript_language)

def update_video_status(db: Session,video: Video,status: TranscriptStatus,last_error: str | None = None,) -> Video:
    video.transcript_status = status
    video.last_error = last_error

    db.commit()
    db.refresh(video)

    return video


def update_video_metadata(
    db: Session,
    video: Video,
    *,
    title: str | None = None,
    transcript_language: str | None = None,
    chunk_count: int | None = None,
    vector_count: int | None = None,
) -> Video:
    if title is not None:
        video.title = title

    if transcript_language is not None:
        video.transcript_language = transcript_language

    if chunk_count is not None:
        video.chunk_count = chunk_count

    if vector_count is not None:
        video.vector_count = vector_count

    db.commit()
    db.refresh(video)

    return video


def mark_video_indexed(
    db: Session,
    video: Video,
    vector_count: int,
) -> Video:
    video.vector_count = vector_count
    video.indexed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(video)

    return video