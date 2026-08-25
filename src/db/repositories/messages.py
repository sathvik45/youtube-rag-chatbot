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
    db: Session,
    thread_id: UUID,
    content: str,
    rewritten_query: str | None = None,
) -> Message:
    message = Message(
        thread_id=thread_id,
        role=MessageRole.USER,
        content=content,
        rewritten_query=rewritten_query,
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    return message


def save_assistant_message(
    db: Session,
    thread_id: UUID,
    content: str,
    *,
    grounded: bool | None = None,
    status: MessageStatus = MessageStatus.ANSWERED,
) -> Message:
    message = Message(
        thread_id=thread_id,
        role=MessageRole.ASSISTANT,
        content=content,
        grounded=grounded,
        status=status,
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    return message


def save_citations(
    db: Session,
    message_id: UUID,
    citations: list[dict],
) -> list[Citation]:
    citation_objects = [
        Citation(
            message_id=message_id,
            video_id=citation["video_id"],
            start_ms=citation["start_ms"],
            end_ms=citation["end_ms"],
            youtube_url=citation["youtube_url"],
            quote_text=citation["quote_text"],
            verified=citation.get("verified", False),
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

    db.add_all(citation_objects)
    db.commit()

    for citation in citation_objects:
        db.refresh(citation)

    return citation_objects


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