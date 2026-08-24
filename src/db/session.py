"""Synchronous SQLAlchemy engine and per-request session helper."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import settings


def _database_url() -> str:
    """Return the configured URL without ever logging credentials."""
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required to use the database")
    return settings.database_url


engine: Engine = create_engine(
    _database_url(),
    pool_pre_ping=True,
    pool_size=3,
    max_overflow=2,
    pool_recycle=1800,
    echo=False,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Yield one session and always return its connection to the pool."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
