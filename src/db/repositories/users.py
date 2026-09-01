from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models.user import User


def create_user(
    session: Session,
    *,
    email: str,
    password_hash: str,
) -> User:
    """Add a user to the current transaction without committing it."""
    user = User(
        email=email,
        password_hash=password_hash,
    )
    session.add(user)
    session.flush()

    return user


def get_user_by_email(
    session: Session,
    *,
    email: str,
) -> User | None:
    statement = select(User).where(User.email == email)

    return session.scalar(statement)


def get_user_by_id(
    session: Session,
    *,
    user_id: UUID,
) -> User | None:
    return session.get(User, user_id)