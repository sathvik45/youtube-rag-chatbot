"""Opt-in PostgreSQL integration tests for persisted thread reads.

These tests use the real API and a rollback-bound database connection.  The
RAG graph is replaced only for the one test that proves a completed chat turn
makes its parent thread recently active.
"""

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.api.deps import get_session_factory, get_thread_message_handler
from src.core.security import create_access_token
from src.db.models.citation import Citation
from src.db.models.message import Message, MessageRole, MessageStatus
from src.db.models.source import Source, SourceStatus, SourceType
from src.db.models.source_video import SourceVideo, SourceVideoStatus
from src.db.models.thread import Thread
from src.db.models.user import User
from src.db.models.video import Video
from src.db.session import engine, get_db
from src.main import app
from src.services.chat import ChatTurn, answer_thread_message


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against PostgreSQL.",
)


@pytest.fixture
def transactional_session_factory():
    """Let request transactions run, then roll every test row back."""
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
    """Ensure every database dependency uses this test's transaction."""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_session_factory] = (
        lambda: transactional_session_factory
    )

    def override_get_db():
        with transactional_session_factory() as session:
            yield session

    # Some read dependencies may use a session directly while others use a
    # factory.  Both must share the same outer rollback transaction.
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def _create_source(
    session: Session,
    *,
    user_id: uuid.UUID,
    suffix: str,
) -> Source:
    source = Source(
        user_id=user_id,
        source_type=SourceType.VIDEO,
        submitted_value=(
            f"https://www.youtube.com/watch?v={suffix[:11]}"
        ),
        status=SourceStatus.READY,
    )
    session.add(source)
    session.flush()
    return source


def _create_thread(
    session: Session,
    *,
    user_id: uuid.UUID,
    source_id: uuid.UUID,
    title: str,
    created_at: datetime,
    updated_at: datetime,
    archived_at: datetime | None = None,
) -> Thread:
    thread = Thread(
        user_id=user_id,
        source_id=source_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        archived_at=archived_at,
    )
    session.add(thread)
    session.flush()
    return thread


def _datetime_from_json(value: str) -> datetime:
    """Accept FastAPI's current +00:00 form as well as a trailing Z."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_list_threads_is_owner_scoped_paginated_and_ordered_by_activity(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex
    shared_time = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    newest_time = shared_time + timedelta(minutes=1)
    hidden_time = newest_time + timedelta(minutes=1)

    with transactional_session_factory() as session:
        with session.begin():
            owner = User(
                email=f"thread-list-owner-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            other_user = User(
                email=f"thread-list-other-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add_all([owner, other_user])
            session.flush()

            newest_source = _create_source(
                session,
                user_id=owner.id,
                suffix=f"{suffix}newest",
            )
            tied_source_one = _create_source(
                session,
                user_id=owner.id,
                suffix=f"{suffix}tiedone",
            )
            tied_source_two = _create_source(
                session,
                user_id=owner.id,
                suffix=f"{suffix}tiedtwo",
            )
            archived_source = _create_source(
                session,
                user_id=owner.id,
                suffix=f"{suffix}archive",
            )
            other_source = _create_source(
                session,
                user_id=other_user.id,
                suffix=f"{suffix}other",
            )

            newest_thread = _create_thread(
                session,
                user_id=owner.id,
                source_id=newest_source.id,
                title="Most recently active",
                created_at=shared_time,
                updated_at=newest_time,
            )
            tied_thread_one = _create_thread(
                session,
                user_id=owner.id,
                source_id=tied_source_one.id,
                title="Tied activity one",
                created_at=shared_time,
                updated_at=shared_time,
            )
            tied_thread_two = _create_thread(
                session,
                user_id=owner.id,
                source_id=tied_source_two.id,
                title="Tied activity two",
                created_at=shared_time,
                updated_at=shared_time,
            )
            archived_thread = _create_thread(
                session,
                user_id=owner.id,
                source_id=archived_source.id,
                title="Archived chat",
                created_at=shared_time,
                updated_at=hidden_time,
                archived_at=hidden_time,
            )
            other_users_thread = _create_thread(
                session,
                user_id=other_user.id,
                source_id=other_source.id,
                title="Another user's newest chat",
                created_at=shared_time,
                updated_at=hidden_time,
            )

            owner_id = owner.id
            tied_threads_descending = sorted(
                [tied_thread_one, tied_thread_two],
                key=lambda thread: thread.id,
                reverse=True,
            )

    headers = {"Authorization": f"Bearer {create_access_token(owner_id)}"}
    first_page_response = transactional_client.get(
        "/threads",
        headers=headers,
        params={"limit": 2},
    )

    assert first_page_response.status_code == 200
    first_page = first_page_response.json()
    first_page_ids = [uuid.UUID(item["id"]) for item in first_page["items"]]

    assert first_page_ids == [
        newest_thread.id,
        tied_threads_descending[0].id,
    ]
    assert first_page["next_cursor"] is not None
    newest_item = first_page["items"][0]
    assert newest_item["id"] == str(newest_thread.id)
    assert newest_item["source_id"] == str(newest_source.id)
    assert newest_item["title"] == "Most recently active"
    assert _datetime_from_json(newest_item["created_at"]) == shared_time
    assert _datetime_from_json(newest_item["updated_at"]) == newest_time

    second_page_response = transactional_client.get(
        "/threads",
        headers=headers,
        params={"limit": 2, "cursor": first_page["next_cursor"]},
    )

    assert second_page_response.status_code == 200
    second_page = second_page_response.json()
    second_page_ids = [
        uuid.UUID(item["id"])
        for item in second_page["items"]
    ]
    assert second_page_ids == [tied_threads_descending[1].id]
    assert second_page["next_cursor"] is None

    returned_ids = first_page_ids + second_page_ids
    assert returned_ids == [
        newest_thread.id,
        tied_threads_descending[0].id,
        tied_threads_descending[1].id,
    ]
    assert archived_thread.id not in returned_ids
    assert other_users_thread.id not in returned_ids


def test_get_thread_messages_is_private_chronological_and_citation_complete(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex
    oldest_time = datetime(2026, 9, 4, 9, 0, tzinfo=timezone.utc)
    tied_time = oldest_time + timedelta(minutes=1)
    newest_time = tied_time + timedelta(minutes=1)

    with transactional_session_factory() as session:
        with session.begin():
            owner = User(
                email=f"thread-history-owner-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            other_user = User(
                email=f"thread-history-other-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add_all([owner, other_user])
            session.flush()

            source = _create_source(
                session,
                user_id=owner.id,
                suffix=f"{suffix}history",
            )
            thread = _create_thread(
                session,
                user_id=owner.id,
                source_id=source.id,
                title="Saved history",
                created_at=oldest_time,
                updated_at=newest_time,
            )
            empty_thread = _create_thread(
                session,
                user_id=owner.id,
                source_id=source.id,
                title="No messages yet",
                created_at=oldest_time,
                updated_at=oldest_time,
            )

            video = Video(
                youtube_video_id=suffix[:11],
                canonical_url=(
                    f"https://www.youtube.com/watch?v={suffix[:11]}"
                ),
                title="Citation video",
            )
            session.add(video)
            session.flush()

            oldest_message = Message(
                thread_id=thread.id,
                role=MessageRole.USER,
                content="First saved question",
                rewritten_query="first standalone question",
                status=MessageStatus.USER_MESSAGE,
                created_at=oldest_time,
            )
            tied_low_message = Message(
                # Explicit UUID values give a deterministic same-timestamp
                # pagination boundary in PostgreSQL.
                id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                thread_id=thread.id,
                role=MessageRole.ASSISTANT,
                content="Earlier tied answer",
                grounded=True,
                status=MessageStatus.ANSWERED,
                created_at=tied_time,
            )
            tied_high_message = Message(
                id=uuid.UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"),
                thread_id=thread.id,
                role=MessageRole.USER,
                content="Later tied question",
                rewritten_query="later standalone question",
                status=MessageStatus.USER_MESSAGE,
                created_at=tied_time,
            )
            newest_message = Message(
                thread_id=thread.id,
                role=MessageRole.ASSISTANT,
                content="No matching context was found.",
                grounded=None,
                status=MessageStatus.REFUSED_NO_CONTEXT,
                created_at=newest_time,
            )
            session.add_all(
                [
                    oldest_message,
                    tied_low_message,
                    tied_high_message,
                    newest_message,
                ]
            )
            session.flush()

            # Insert out of display order.  The read API must expose citation
            # position, not insertion order, so the client can render proof
            # chips consistently after a page reload.
            session.add_all(
                [
                    Citation(
                        message_id=tied_low_message.id,
                        video_id=video.id,
                        start_ms=8_000,
                        end_ms=10_000,
                        youtube_url=video.canonical_url,
                        quote_text="second displayed quote",
                        verified=None,
                        verification_score=None,
                        position=2,
                    ),
                    Citation(
                        message_id=tied_low_message.id,
                        video_id=video.id,
                        start_ms=2_000,
                        end_ms=5_000,
                        youtube_url=video.canonical_url,
                        quote_text="first displayed quote",
                        verified=True,
                        verification_score=0.93,
                        position=1,
                    ),
                ]
            )

            other_source = _create_source(
                session,
                user_id=other_user.id,
                suffix=f"{suffix}private",
            )
            _create_thread(
                session,
                user_id=other_user.id,
                source_id=other_source.id,
                title="Other private history",
                created_at=oldest_time,
                updated_at=newest_time,
            )

            owner_id = owner.id
            other_user_id = other_user.id
            thread_id = thread.id
            empty_thread_id = empty_thread.id

    owner_headers = {
        "Authorization": f"Bearer {create_access_token(owner_id)}"
    }
    first_page_response = transactional_client.get(
        f"/threads/{thread_id}/messages",
        headers=owner_headers,
        params={"limit": 2},
    )

    assert first_page_response.status_code == 200
    first_page = first_page_response.json()
    assert first_page["thread_id"] == str(thread_id)
    assert [uuid.UUID(item["id"]) for item in first_page["items"]] == [
        tied_high_message.id,
        newest_message.id,
    ]
    assert [item["created_at"] for item in first_page["items"]] == sorted(
        item["created_at"] for item in first_page["items"]
    )
    assert first_page["items"][0]["role"] == "user"
    assert first_page["items"][0]["rewritten_query"] == (
        "later standalone question"
    )
    assert first_page["items"][0]["citations"] == []
    assert first_page["items"][1]["status"] == "refused_no_context"
    assert first_page["items"][1]["grounded"] is None
    assert first_page["items"][1]["citations"] == []
    assert first_page["next_cursor"] is not None

    second_page_response = transactional_client.get(
        f"/threads/{thread_id}/messages",
        headers=owner_headers,
        params={"limit": 2, "cursor": first_page["next_cursor"]},
    )

    assert second_page_response.status_code == 200
    second_page = second_page_response.json()
    assert [uuid.UUID(item["id"]) for item in second_page["items"]] == [
        oldest_message.id,
        tied_low_message.id,
    ]
    assert second_page["next_cursor"] is None

    returned_in_chronological_order = [
        uuid.UUID(item["id"])
        for item in second_page["items"] + first_page["items"]
    ]
    assert returned_in_chronological_order == [
        oldest_message.id,
        tied_low_message.id,
        tied_high_message.id,
        newest_message.id,
    ]

    oldest_item, cited_answer = second_page["items"]
    assert oldest_item["rewritten_query"] == "first standalone question"
    assert cited_answer["role"] == "assistant"
    assert cited_answer["status"] == "answered"
    assert cited_answer["grounded"] is True
    assert [citation["position"] for citation in cited_answer["citations"]] == [
        1,
        2,
    ]
    assert cited_answer["citations"] == [
        {
            "id": cited_answer["citations"][0]["id"],
            "video_id": str(video.id),
            "start_ms": 2_000,
            "end_ms": 5_000,
            "youtube_url": video.canonical_url,
            "quote_text": "first displayed quote",
            "verified": True,
            "verification_score": 0.93,
            "position": 1,
        },
        {
            "id": cited_answer["citations"][1]["id"],
            "video_id": str(video.id),
            "start_ms": 8_000,
            "end_ms": 10_000,
            "youtube_url": video.canonical_url,
            "quote_text": "second displayed quote",
            "verified": None,
            "verification_score": None,
            "position": 2,
        },
    ]

    empty_response = transactional_client.get(
        f"/threads/{empty_thread_id}/messages",
        headers=owner_headers,
    )
    assert empty_response.status_code == 200
    assert empty_response.json() == {
        "thread_id": str(empty_thread_id),
        "items": [],
        "next_cursor": None,
    }

    forbidden_response = transactional_client.get(
        f"/threads/{thread_id}/messages",
        headers={
            "Authorization": f"Bearer {create_access_token(other_user_id)}"
        },
    )
    assert forbidden_response.status_code == 404
    assert forbidden_response.json() == {"detail": "Thread not found."}


@dataclass
class StaticAnswerGraph:
    """Small deterministic graph used to exercise the real POST route."""

    answer: str = "The uploaded video explains this topic."

    async def ainvoke(
        self,
        state: dict[str, Any],
        *,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        del config
        return {
            "messages": [*state["messages"], AIMessage(content=self.answer)],
            "query": "What does the video explain?",
            "outcome": "answered",
            "grounded": True,
            "citations": [],
        }


class FailingGraph:
    """Deterministic provider failure for the durable error-path test."""

    async def ainvoke(
        self,
        state: dict[str, Any],
        *,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        del state, config
        raise RuntimeError("test-only provider outage")


def test_post_message_updates_parent_thread_activity_for_the_chat_sidebar(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex
    old_time = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(
        days=1
    )
    newer_before_post = old_time + timedelta(minutes=1)

    with transactional_session_factory() as session:
        with session.begin():
            owner = User(
                email=f"thread-activity-owner-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(owner)
            session.flush()

            target_source = _create_source(
                session,
                user_id=owner.id,
                suffix=f"{suffix}target",
            )
            target_video = Video(
                youtube_video_id=suffix[:11],
                canonical_url=(
                    f"https://www.youtube.com/watch?v={suffix[:11]}"
                ),
                title="Ready video for the chat turn",
            )
            session.add(target_video)
            session.flush()
            session.add(
                SourceVideo(
                    source_id=target_source.id,
                    video_id=target_video.id,
                    position=0,
                    status=SourceVideoStatus.READY,
                )
            )

            target_thread = _create_thread(
                session,
                user_id=owner.id,
                source_id=target_source.id,
                title="Will become active",
                created_at=old_time,
                updated_at=old_time,
            )
            other_source = _create_source(
                session,
                user_id=owner.id,
                suffix=f"{suffix}other",
            )
            other_thread = _create_thread(
                session,
                user_id=owner.id,
                source_id=other_source.id,
                title="Was active before the turn",
                created_at=old_time,
                updated_at=newer_before_post,
            )

            owner_id = owner.id
            target_thread_id = target_thread.id

    graph = StaticAnswerGraph()

    async def handle_message(
        user_id: uuid.UUID,
        thread_id: uuid.UUID,
        content: str,
    ) -> ChatTurn:
        return await answer_thread_message(
            user_id,
            thread_id,
            content,
            session_factory=transactional_session_factory,
            graph=graph,
        )

    app.dependency_overrides[get_thread_message_handler] = lambda: handle_message
    headers = {"Authorization": f"Bearer {create_access_token(owner_id)}"}

    post_response = transactional_client.post(
        f"/threads/{target_thread_id}/messages",
        headers=headers,
        json={"content": "What does this video explain?"},
    )
    assert post_response.status_code == 200

    list_response = transactional_client.get(
        "/threads",
        headers=headers,
        params={"limit": 2},
    )
    assert list_response.status_code == 200

    listed_threads = list_response.json()["items"]
    assert [uuid.UUID(item["id"]) for item in listed_threads] == [
        target_thread_id,
        other_thread.id,
    ]
    assert _datetime_from_json(listed_threads[0]["updated_at"]) > old_time


def test_temporary_chat_error_also_updates_parent_thread_activity(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex
    old_time = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(
        days=1
    )

    with transactional_session_factory() as session:
        with session.begin():
            owner = User(
                email=f"thread-error-owner-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(owner)
            session.flush()

            source = _create_source(
                session,
                user_id=owner.id,
                suffix=f"{suffix}error",
            )
            video = Video(
                youtube_video_id=suffix[:11],
                canonical_url=(
                    f"https://www.youtube.com/watch?v={suffix[:11]}"
                ),
                title="Ready video for the failed chat turn",
            )
            session.add(video)
            session.flush()
            session.add(
                SourceVideo(
                    source_id=source.id,
                    video_id=video.id,
                    position=0,
                    status=SourceVideoStatus.READY,
                )
            )
            thread = _create_thread(
                session,
                user_id=owner.id,
                source_id=source.id,
                title="Error still counts as activity",
                created_at=old_time,
                updated_at=old_time,
            )

            owner_id = owner.id
            thread_id = thread.id

    async def handle_message(
        user_id: uuid.UUID,
        received_thread_id: uuid.UUID,
        content: str,
    ) -> ChatTurn:
        return await answer_thread_message(
            user_id,
            received_thread_id,
            content,
            session_factory=transactional_session_factory,
            graph=FailingGraph(),
        )

    app.dependency_overrides[get_thread_message_handler] = lambda: handle_message
    response = transactional_client.post(
        f"/threads/{thread_id}/messages",
        headers={"Authorization": f"Bearer {create_access_token(owner_id)}"},
        json={"content": "What does the video explain?"},
    )

    assert response.status_code == 503

    with transactional_session_factory() as session:
        thread_after_error = session.get(Thread, thread_id)
        temporary_error = session.scalar(
            select(Message).where(
                Message.thread_id == thread_id,
                Message.status == MessageStatus.TEMPORARY_ERROR,
            )
        )

    assert thread_after_error is not None
    assert thread_after_error.updated_at > old_time
    assert temporary_error is not None
