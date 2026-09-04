from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

from src.db.models.citation import Citation
from src.db.models.message import MessageRole, MessageStatus
from src.db.models.source import SourceStatus
from src.db.repositories.messages import (
    list_citations_for_messages,
    list_messages_for_thread,
)
from src.db.repositories.sources import get_source_progress
from src.db.repositories.threads import (
    create_thread,
    get_thread_for_user,
    list_user_threads,
)
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


@dataclass(frozen=True)
class ListedThread:
    id: UUID
    source_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ThreadMessageCitation:
    id: UUID
    video_id: UUID
    start_ms: int
    end_ms: int
    youtube_url: str
    quote_text: str | None
    verified: bool | None
    verification_score: float | None
    position: int


@dataclass(frozen=True)
class ThreadHistoryMessage:
    id: UUID
    role: MessageRole
    content: str
    rewritten_query: str | None
    grounded: bool | None
    status: MessageStatus
    created_at: datetime
    citations: tuple[ThreadMessageCitation, ...]


@dataclass(frozen=True)
class ThreadHistory:
    """A raw newest-first page; the route reverses the visible response."""

    thread_id: UUID
    messages: tuple[ThreadHistoryMessage, ...]


def _citation_view(citation: Citation) -> ThreadMessageCitation:
    return ThreadMessageCitation(
        id=citation.id,
        video_id=citation.video_id,
        start_ms=citation.start_ms,
        end_ms=citation.end_ms,
        youtube_url=citation.youtube_url,
        quote_text=citation.quote_text,
        verified=citation.verified,
        verification_score=citation.verification_score,
        position=citation.position,
    )


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


def list_threads_for_user(
    user_id: UUID,
    *,
    limit: int,
    before_updated_at: datetime | None = None,
    before_id: UUID | None = None,
    session_factory: SessionFactory = SessionLocal,
) -> list[ListedThread]:
    """Read one tie-breaker-safe newest-active page from a live sidebar.

    Activity timestamps are intentionally mutable.  A client receiving a new
    chat event should refresh its sidebar rather than treating a long cursor
    walk as a historical snapshot.
    """
    session = session_factory()
    try:
        with session.begin():
            threads = list_user_threads(
                session,
                user_id=user_id,
                limit=limit,
                before_updated_at=before_updated_at,
                before_id=before_id,
            )

            return [
                ListedThread(
                    id=thread.id,
                    source_id=thread.source_id,
                    title=thread.title,
                    created_at=thread.created_at,
                    updated_at=thread.updated_at,
                )
                for thread in threads
            ]
    finally:
        session.close()


def get_thread_history_for_user(
    user_id: UUID,
    thread_id: UUID,
    *,
    limit: int,
    before_created_at: datetime | None = None,
    before_id: UUID | None = None,
    session_factory: SessionFactory = SessionLocal,
) -> ThreadHistory:
    """Load an owned thread's recovery page and its nested citations.

    The repository returns newest-first records so a client initially sees the
    latest part of a long conversation.  The route reverses only the visible
    page before returning it, retaining chronological display order.
    """
    session = session_factory()
    try:
        with session.begin():
            thread = get_thread_for_user(session, thread_id, user_id)
            if thread is None:
                raise LookupError(f"Thread {thread_id} was not found.")

            messages = list_messages_for_thread(
                session,
                thread_id=thread.id,
                limit=limit,
                before_created_at=before_created_at,
                before_id=before_id,
            )
            citations_by_message = list_citations_for_messages(
                session,
                message_ids=[message.id for message in messages],
            )

            return ThreadHistory(
                thread_id=thread.id,
                messages=tuple(
                    ThreadHistoryMessage(
                        id=message.id,
                        role=message.role,
                        content=message.content,
                        rewritten_query=message.rewritten_query,
                        grounded=message.grounded,
                        status=message.status,
                        created_at=message.created_at,
                        citations=tuple(
                            _citation_view(citation)
                            for citation in citations_by_message.get(
                                message.id,
                                [],
                            )
                        ),
                    )
                    for message in messages
                ),
            )
    finally:
        session.close()
