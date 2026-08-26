from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models.source_video import SourceVideo, SourceVideoStatus


def update_source_video_status(
    session: Session,
    *,
    source_id: UUID,
    video_id: UUID,
    status: SourceVideoStatus,
    error_message: str | None = None,
) -> SourceVideo:
    """Update ingestion state for one video within one submitted source."""
    source_video = session.scalar(
        select(SourceVideo).where(
            SourceVideo.source_id == source_id,
            SourceVideo.video_id == video_id,
        )
    )

    if source_video is None:
        raise LookupError(
            f"No source/video link exists for source {source_id} and video {video_id}."
        )

    source_video.status = status
    source_video.error_message = error_message
    session.flush()

    return source_video