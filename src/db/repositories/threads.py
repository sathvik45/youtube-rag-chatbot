from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from src.db.models.thread import Thread


def create_thread(
    session: Session,
    *,
    user_id: UUID,
    source_id: UUID,
    title: str,
) -> Thread:
    """Add a thread to the caller's transaction without committing it."""
    thread = Thread(
        user_id=user_id,
        source_id=source_id,
        title=title,
    )

    session.add(thread)
    session.flush()

    return thread


def get_user_threads(
    db: Session,
    user_id: UUID,
) -> list[Thread]:
    statement = select(Thread).where(Thread.user_id == user_id).order_by(
        Thread.updated_at.desc(), Thread.id.desc()
    )

    return list(db.scalars(statement).all())


def list_user_threads(
    session: Session,
    *,
    user_id: UUID,
    limit: int,
    before_updated_at: datetime | None = None,
    before_id: UUID | None = None,
) -> list[Thread]:
    """Return one stable newest-first page of a user's chats."""
    if (before_updated_at is None) != (before_id is None):
        raise ValueError(
            "before_updated_at and before_id must be provided together."
        )

    statement = (
        select(Thread)
        .where(Thread.user_id == user_id)
        .order_by(Thread.updated_at.desc(), Thread.id.desc())
        .limit(limit)
    )

    if before_updated_at is not None and before_id is not None:
        statement = statement.where(
            or_(
                Thread.updated_at < before_updated_at,
                and_(
                    Thread.updated_at == before_updated_at,
                    Thread.id < before_id,
                ),
            )
        )

    return list(session.scalars(statement))


def delete_thread(
    session: Session,
    *,
    thread: Thread,
) -> None:
    """Delete one loaded thread; PostgreSQL cascades to its messages/citations."""
    session.delete(thread)
    session.flush()


def get_thread_for_user(
    db: Session,
    thread_id: UUID,
    user_id: UUID,
    *,
    for_update: bool = False,
) -> Thread | None:
    statement = (
        select(Thread)
        .where(
            Thread.id == thread_id,
            Thread.user_id == user_id,
        )
    )

    if for_update:
        statement = statement.with_for_update()

    return db.scalar(statement)


def touch_thread(
    session: Session,
    *,
    thread_id: UUID,
    now: datetime | None = None,
) -> Thread:
    """Record chat activity explicitly when a child message is persisted."""
    thread = session.get(Thread, thread_id)
    if thread is None:
        raise LookupError(f"Thread {thread_id} was not found.")

    thread.updated_at = now or datetime.now(timezone.utc)
    session.flush()

    return thread
