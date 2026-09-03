from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

from src.db.models.source import SourceStatus
from src.db.repositories.sources import get_source_progress
from src.db.repositories.threads import create_thread
from src.db.session import SessionLocal


SessionFactory = Callable[[], Session]


class SourceNotReadyForChatError(ValueError):
    """Raised when a source has no indexed video content to answer from."""


@dataclass(frozen=True)
class CreatedThread:
    id: UUID
    source_id: UUID
    title: str
    created_at: datetime


def create_thread_for_source(
    user_id: UUID,
    source_id: UUID,
    title: str,
    *,
    session_factory: SessionFactory = SessionLocal,
) -> CreatedThread:
    """Create one user-owned chat thread scoped to a chat-ready source."""
    session = session_factory()
    try:
        with session.begin():
            progress = get_source_progress(
                session,
                source_id=source_id,
                user_id=user_id,
            )

            if progress.source_status not in {
                SourceStatus.READY,
                SourceStatus.PARTIAL,
            }:
                raise SourceNotReadyForChatError(
                    "Source is not ready for chat yet."
                )

            if progress.video_counts["ready"] == 0:
                raise SourceNotReadyForChatError(
                    "Source has no ready videos to use for chat."
                )

            thread = create_thread(
                session,
                user_id=user_id,
                source_id=source_id,
                title=title,
            )

            return CreatedThread(
                id=thread.id,
                source_id=thread.source_id,
                title=thread.title,
                created_at=thread.created_at,
            )
    finally:
        session.close()
