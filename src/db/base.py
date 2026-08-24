from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared registry for every SQLAlchemy model in the application."""

    pass
