"""Opt-in integration tests for JWT-based source ownership."""

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from src.core.security import create_access_token
from src.db.models.source import Source, SourceStatus, SourceType
from src.db.models.user import User
from src.db.repositories.sources import create_pending_source
from src.db.session import engine, get_db
from src.main import app

from src.api.deps import get_session_factory

from datetime import datetime, timedelta, timezone

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against PostgreSQL.",
)


@pytest.fixture
def transactional_session_factory():
    """Let API request sessions run, then discard all database rows."""
    connection = engine.connect()
    outer_transaction = connection.begin()

    factory = sessionmaker(
        bind=connection,
        autoflush=False,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    try:
        yield factory
    finally:
        outer_transaction.rollback()
        connection.close()


@pytest.fixture
def transactional_client(transactional_session_factory):
    """Make FastAPI use sessions bound to this test's outer transaction."""
    app.dependency_overrides.clear()

    def override_get_db():
        with transactional_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_session_factory] = (
        lambda: transactional_session_factory
    )
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def test_source_status_requires_ownership(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex

    with transactional_session_factory() as session:
        with session.begin():
            owner = User(
                email=f"owner-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            other_user = User(
                email=f"other-user-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add_all([owner, other_user])
            session.flush()

            source = create_pending_source(
                session,
                user_id=owner.id,
                source_type=SourceType.VIDEO,
                submitted_value=(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                ),
            )

            owner_id = owner.id
            other_user_id = other_user.id
            source_id = source.id

    owner_token = create_access_token(owner_id)
    other_user_token = create_access_token(other_user_id)

    owner_response = transactional_client.get(
        f"/sources/{source_id}",
        headers={"Authorization": f"Bearer {owner_token}"},
    )

    assert owner_response.status_code == 200
    assert owner_response.json()["source_id"] == str(source_id)
    assert owner_response.json()["status"] == "pending"
    assert owner_response.json()["total_videos"] == 0
    assert owner_response.json()["completed_videos"] == 0

    other_user_response = transactional_client.get(
        f"/sources/{source_id}",
        headers={"Authorization": f"Bearer {other_user_token}"},
    )

    assert other_user_response.status_code == 404
    assert other_user_response.json() == {
        "detail": "Source not found."
    }

def test_list_sources_is_paginated_and_owner_scoped(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex
    shared_created_at = datetime.now(timezone.utc).replace(
        microsecond=0
    )
    newest_created_at = shared_created_at + timedelta(minutes=1)
    other_user_created_at = shared_created_at + timedelta(minutes=2)

    with transactional_session_factory() as session:
        with session.begin():
            owner = User(
                email=f"list-owner-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            other_user = User(
                email=f"list-other-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add_all([owner, other_user])
            session.flush()

            newest_source = Source(
                user_id=owner.id,
                source_type=SourceType.PLAYLIST,
                submitted_value=(
                    "https://www.youtube.com/playlist?list=PLnewest"
                ),
                status=SourceStatus.PROCESSING,
                created_at=newest_created_at,
            )
            tied_source_one = Source(
                user_id=owner.id,
                source_type=SourceType.VIDEO,
                submitted_value=(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                ),
                status=SourceStatus.PENDING,
                created_at=shared_created_at,
            )
            tied_source_two = Source(
                user_id=owner.id,
                source_type=SourceType.VIDEO,
                submitted_value=(
                    "https://www.youtube.com/watch?v=9bZkp7q19f0"
                ),
                status=SourceStatus.READY,
                created_at=shared_created_at,
            )
            other_user_source = Source(
                user_id=other_user.id,
                source_type=SourceType.VIDEO,
                submitted_value=(
                    "https://www.youtube.com/watch?v=otheruser01"
                ),
                status=SourceStatus.READY,
                created_at=other_user_created_at,
            )
            session.add_all(
                [
                    newest_source,
                    tied_source_one,
                    tied_source_two,
                    other_user_source,
                ]
            )
            session.flush()

            owner_id = owner.id

            tied_sources_descending = sorted(
                [tied_source_one, tied_source_two],
                key=lambda source: source.id,
                reverse=True,
            )

    token = create_access_token(owner_id)
    headers = {"Authorization": f"Bearer {token}"}

    first_page_response = transactional_client.get(
        "/sources",
        headers=headers,
        params={"limit": 2},
    )

    assert first_page_response.status_code == 200

    first_page = first_page_response.json()
    first_page_ids = [
        uuid.UUID(item["id"])
        for item in first_page["items"]
    ]

    assert first_page_ids == [
        newest_source.id,
        tied_sources_descending[0].id,
    ]
    assert first_page["next_cursor"] is not None

    second_page_response = transactional_client.get(
        "/sources",
        headers=headers,
        params={
            "limit": 2,
            "cursor": first_page["next_cursor"],
        },
    )

    assert second_page_response.status_code == 200

    second_page = second_page_response.json()
    second_page_ids = [
        uuid.UUID(item["id"])
        for item in second_page["items"]
    ]

    assert second_page_ids == [tied_sources_descending[1].id]
    assert second_page["next_cursor"] is None

    returned_ids = first_page_ids + second_page_ids

    assert returned_ids == [
        newest_source.id,
        tied_sources_descending[0].id,
        tied_sources_descending[1].id,
    ]
    assert len(returned_ids) == len(set(returned_ids))
    assert other_user_source.id not in returned_ids