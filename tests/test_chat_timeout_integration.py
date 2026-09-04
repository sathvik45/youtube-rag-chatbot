"""Opt-in integration coverage for bounded thread-chat RAG calls.

The graph replacement intentionally sleeps longer than the injected timeout.
This proves that the route returns its safe retryable response while the real
message service still records one durable temporary-error assistant turn.
"""

import asyncio
import os
import uuid
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
from src.db.models.source import SourceStatus, SourceType
from src.db.models.source_video import SourceVideoStatus
from src.db.models.thread import Thread
from src.db.models.user import User
from src.db.repositories.sources import attach_videos, create_pending_source
from src.db.repositories.videos import get_or_create_video
from src.db.session import engine
from src.main import app
from src.services.chat import ChatTurn, answer_thread_message


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run against PostgreSQL.",
)


@pytest.fixture
def transactional_session_factory():
    """Allow request transactions while rolling back all test rows at the end."""
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
    """Make every request use this test's rollback-bound database factory."""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_session_factory] = (
        lambda: transactional_session_factory
    )

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


class SlowGraph:
    """A valid graph response that arrives after the service deadline."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(
        self,
        state: dict[str, Any],
        *,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        del config
        self.calls += 1
        await asyncio.sleep(1)
        return {
            "messages": [
                *state["messages"],
                AIMessage(content="This answer arrived too late."),
            ],
            "query": "What does the video explain?",
            "outcome": "answered",
            "grounded": True,
            "citations": [],
        }


def _create_ready_thread(session: Session, *, owner: User) -> Thread:
    """Seed the smallest source-backed thread that is eligible for chat."""
    youtube_video_id = uuid.uuid4().hex[:11]
    source = create_pending_source(
        session,
        user_id=owner.id,
        source_type=SourceType.VIDEO,
        submitted_value=(
            f"https://www.youtube.com/watch?v={youtube_video_id}"
        ),
    )
    source.status = SourceStatus.READY

    video = get_or_create_video(
        session,
        youtube_video_id=youtube_video_id,
        canonical_url=(
            f"https://www.youtube.com/watch?v={youtube_video_id}"
        ),
        title="Ready video for timeout test",
    )
    source_video = attach_videos(
        session,
        source_id=source.id,
        video_ids=[video.id],
    )[0]
    source_video.status = SourceVideoStatus.READY

    thread = Thread(
        user_id=owner.id,
        source_id=source.id,
        title="Timeout-safe thread",
    )
    session.add(thread)
    session.flush()
    return thread


def test_slow_graph_returns_503_and_persists_one_temporary_error_turn(
    transactional_session_factory,
    transactional_client,
    caplog,
) -> None:
    """A timeout must not create citations or expose a late graph response."""
    suffix = uuid.uuid4().hex
    with transactional_session_factory() as session:
        with session.begin():
            owner = User(
                email=f"timeout-owner-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(owner)
            session.flush()

            thread = _create_ready_thread(session, owner=owner)
            owner_id = owner.id
            thread_id = thread.id

    graph = SlowGraph()

    async def handle(
        user_id: uuid.UUID,
        received_thread_id: uuid.UUID,
        content: str,
    ) -> ChatTurn:
        return await answer_thread_message(
            user_id,
            received_thread_id,
            content,
            session_factory=transactional_session_factory,
            graph=graph,
            rag_timeout_seconds=0.01,
        )

    # The route uses the real chat service, but this dependency injection keeps
    # the test completely isolated from live RAG providers.
    app.dependency_overrides[get_thread_message_handler] = lambda: handle
    question = "What does the video explain?"
    with caplog.at_level("WARNING", logger="src.services.chat"):
        response = transactional_client.post(
            f"/threads/{thread_id}/messages",
            headers={"Authorization": f"Bearer {create_access_token(owner_id)}"},
            json={"content": question},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Chat is temporarily unavailable. Please try again."
    }
    assert graph.calls == 1

    with transactional_session_factory() as session:
        messages = list(
            session.scalars(
                select(Message)
                .where(Message.thread_id == thread_id)
                .order_by(Message.created_at, Message.id)
            )
        )
        citations = list(
            session.scalars(
                select(Citation)
                .join(Message, Citation.message_id == Message.id)
                .where(Message.thread_id == thread_id)
            )
        )

    assert [message.role for message in messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    temporary_errors = [
        message
        for message in messages
        if message.status is MessageStatus.TEMPORARY_ERROR
    ]
    assert len(temporary_errors) == 1
    assert temporary_errors[0].role is MessageRole.ASSISTANT
    assert citations == []

    failure_logs = [
        record.getMessage()
        for record in caplog.records
        if record.name == "src.services.chat"
        and record.getMessage().startswith("chat_turn_failed ")
    ]
    assert len(failure_logs) == 1
    assert "failure_type=timeout" in failure_logs[0]
    assert "thread_id=" in failure_logs[0]
    assert "rag_duration_ms=" in failure_logs[0]
    assert question not in failure_logs[0]
    assert "This answer arrived too late." not in failure_logs[0]
