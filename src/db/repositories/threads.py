from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.thread import Thread


def create_thread(
    db: Session,
    user_id: UUID,
    source_id: UUID,
    title: str,
) -> Thread:
    thread = Thread(
        user_id=user_id,
        source_id=source_id,
        title=title,
    )

    db.add(thread)
    db.commit()
    db.refresh(thread)

    return thread


def get_user_threads(
    db: Session,
    user_id: UUID,
    *,
    include_archived: bool = False,
) -> list[Thread]:
    statement = (
        select(Thread)
        .where(Thread.user_id == user_id)
        .order_by(Thread.updated_at.desc())
    )

    if not include_archived:
        statement = statement.where(
            Thread.archived_at.is_(None)
        )

    return list(db.scalars(statement).all())


def get_thread_for_user(
    db: Session,
    thread_id: UUID,
    user_id: UUID,
) -> Thread | None:
    statement = (
        select(Thread)
        .where(
            Thread.id == thread_id,
            Thread.user_id == user_id,
        )
    )

    return db.scalar(statement)