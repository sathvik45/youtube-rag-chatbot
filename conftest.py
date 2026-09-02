"""Fail-closed safety guard for database integration tests."""

from __future__ import annotations

import os
from ipaddress import ip_address

import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from src.core.config import settings


LOCAL_TEST_DATABASE = "rag_eval_test"


def _is_literal_loopback(host: str | None) -> bool:
    """Return True only for a literal loopback IP address."""
    if host is None:
        return False

    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def pytest_configure(config: pytest.Config) -> None:
    """Reject integration tests unless they target local PostgreSQL."""
    if os.getenv("RUN_DB_INTEGRATION") != "1":
        return

    if not settings.database_url:
        raise pytest.UsageError(
            "RUN_DB_INTEGRATION=1 requires a local PostgreSQL "
            "test DATABASE_URL."
        )

    try:
        database_url = make_url(settings.database_url)
    except ArgumentError:
        raise pytest.UsageError(
            "RUN_DB_INTEGRATION=1 requires a valid local PostgreSQL "
            "test DATABASE_URL."
        ) from None

    if database_url.query:
        raise pytest.UsageError(
            "Refusing database integration tests. The local test "
            "DATABASE_URL must not contain query parameters."
        )

    is_safe_test_database = (
        database_url.get_backend_name() == "postgresql"
        and _is_literal_loopback(database_url.host)
        and database_url.database == LOCAL_TEST_DATABASE
    )

    if not is_safe_test_database:
        raise pytest.UsageError(
            "Refusing database integration tests. Use local PostgreSQL on "
            "a literal loopback address such as 127.0.0.1, with database "
            "name rag_eval_test."
        )