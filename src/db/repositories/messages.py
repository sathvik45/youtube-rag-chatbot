from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models.citation import Citation
from src.db.models.message import (
    Message,
    MessageRole,
    MessageStatus,
)


def save_user_message(
    session: Session,
    *,
    thread_id: UUID,
    content: str,
    rewritten_query: str | None = None,
) -> Message:
    """Add one user message to the caller's transaction."""
    message = Message(
        thread_id=thread_id,
        role=MessageRole.USER,
        content=content,
        rewritten_query=rewritten_query,
    )

    session.add(message)
    session.flush()

    return message


def save_assistant_message(
    session: Session,
    *,
    thread_id: UUID,
    content: str,
    grounded: bool | None = None,
    status: MessageStatus = MessageStatus.ANSWERED,
) -> Message:
    """Add one assistant message to the caller's transaction."""
    message = Message(
        thread_id=thread_id,
        role=MessageRole.ASSISTANT,
        content=content,
        grounded=grounded,
        status=status,
    )

    session.add(message)
    session.flush()

    return message


def save_citations(
    session: Session,
    *,
    message_id: UUID,
    citations: list[dict],
) -> list[Citation]:
    """Add ordered citations to the caller's transaction."""
    citation_objects = [
        Citation(
            message_id=message_id,
            video_id=citation["video_id"],
            start_ms=citation["start_ms"],
            end_ms=citation["end_ms"],
            youtube_url=citation["youtube_url"],
            quote_text=citation.get("quote_text"),
            verified=citation.get("verified"),
            verification_score=citation.get(
                "verification_score"
            ),
            position=position,
        )
        for position, citation in enumerate(
            citations,
            start=1,
        )
    ]

    session.add_all(citation_objects)
    session.flush()

    return citation_objects


def set_user_message_rewritten_query(
    session: Session,
    *,
    message_id: UUID,
    rewritten_query: str | None,
) -> Message:
    """Store the graph's standalone retrieval query for one user message."""
    message = session.get(Message, message_id)

    if message is None:
        raise LookupError(f"Message {message_id} was not found.")

    if message.role is not MessageRole.USER:
        raise ValueError(f"Message {message_id} is not a user message.")

    message.rewritten_query = rewritten_query
    session.flush()

    return message


def get_recent_messages(
    db: Session,
    thread_id: UUID,
    limit: int = 20,
) -> list[Message]:
    statement = (
        select(Message)
        .where(Message.thread_id == thread_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )

    messages = list(db.scalars(statement).all())

    messages.reverse()

    return messages
