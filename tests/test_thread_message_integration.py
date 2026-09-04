"""Opt-in integration tests for thread-scoped RAG messages.

The graph is deliberately replaced with a recording fake.  These tests still
exercise the real message service, database transactions, ownership checks,
and FastAPI route; they just do not need an LLM, embeddings, or Pinecone.
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage
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


@dataclass
class GraphCall:
    messages: list[BaseMessage]
    video_ids: list[str]


@dataclass
class RecordingGraph:
    """A deterministic graph replacement that records its private scope."""

    answer: str
    query: str
    outcome: str
    grounded: bool | None
    citations: list[dict[str, Any]] = field(default_factory=list)
    calls: list[GraphCall] = field(default_factory=list)

    async def ainvoke(
        self,
        state: dict[str, Any],
        *,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        messages = list(state["messages"])
        video_ids = list(config["configurable"]["video_ids"])
        self.calls.append(GraphCall(messages=messages, video_ids=video_ids))

        return {
            "messages": [*messages, AIMessage(content=self.answer)],
            "query": self.query,
            "outcome": self.outcome,
            "grounded": self.grounded,
            "citations": self.citations,
        }


def _install_real_message_handler(
    transactional_session_factory,
    graph: RecordingGraph,
) -> None:
    """Wire the real service into the route, with only RAG made deterministic."""

    async def handle(
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

    app.dependency_overrides[get_thread_message_handler] = lambda: handle


def _create_thread_source(
    session: Session,
    *,
    owner: User,
    ready_youtube_video_id: str | None,
    source_status: SourceStatus,
) -> tuple[Thread, uuid.UUID | None]:
    """Seed a source-linked thread, optionally with one ready video."""
    source = create_pending_source(
        session,
        user_id=owner.id,
        source_type=SourceType.VIDEO,
        submitted_value="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )
    source.status = source_status

    ready_video_id: uuid.UUID | None = None
    if ready_youtube_video_id is not None:
        video = get_or_create_video(
            session,
            youtube_video_id=ready_youtube_video_id,
            canonical_url=(
                f"https://www.youtube.com/watch?v={ready_youtube_video_id}"
            ),
            title="Ready scoped video",
        )
        source_video = attach_videos(
            session,
            source_id=source.id,
            video_ids=[video.id],
        )[0]
        source_video.status = SourceVideoStatus.READY
        ready_video_id = video.id

    thread = Thread(
        user_id=owner.id,
        source_id=source.id,
        title="Thread-scoped chat",
    )
    session.add(thread)
    session.flush()

    return thread, ready_video_id


def test_post_message_uses_only_ready_thread_videos_and_persists_citations(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex
    ready_youtube_video_id = suffix[:11]
    not_ready_youtube_video_id = suffix[11:22]

    with transactional_session_factory() as session:
        with session.begin():
            owner = User(
                email=f"message-owner-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(owner)
            session.flush()

            thread, ready_video_id = _create_thread_source(
                session,
                owner=owner,
                ready_youtube_video_id=ready_youtube_video_id,
                source_status=SourceStatus.PARTIAL,
            )
            assert ready_video_id is not None

            # A partial playlist/source must not leak an unfinished video's
            # vectors into the graph scope.
            unavailable_video = get_or_create_video(
                session,
                youtube_video_id=not_ready_youtube_video_id,
                canonical_url=(
                    f"https://www.youtube.com/watch?v={not_ready_youtube_video_id}"
                ),
                title="Not ready scoped video",
            )
            # attach_videos() numbers each supplied batch from zero.  This is
            # a second link, so seed its real playlist position explicitly.
            unavailable_link = SourceVideo(
                source_id=thread.source_id,
                video_id=unavailable_video.id,
                position=1,
                status=SourceVideoStatus.PROCESSING,
            )
            session.add(unavailable_link)

            # A message in a different thread must never become part of this
            # thread's graph history, even for the same authenticated owner.
            other_thread, _ = _create_thread_source(
                session,
                owner=owner,
                ready_youtube_video_id=uuid.uuid4().hex[:11],
                source_status=SourceStatus.READY,
            )
            session.add(
                Message(
                    thread_id=other_thread.id,
                    role=MessageRole.USER,
                    content="Private history from another thread.",
                )
            )
            session.flush()

            owner_id = owner.id
            thread_id = thread.id

    graph = RecordingGraph(
        answer='The ready video says this. [c1: "ready evidence"]',
        query="What does the ready video say?",
        outcome="answered",
        grounded=True,
        citations=[
            {
                "video_id": ready_youtube_video_id,
                "start_ms": 1_000,
                "end_ms": 4_500,
                "url": (
                    "https://www.youtube.com/watch?"
                    f"v={ready_youtube_video_id}&t=0s"
                ),
                "quotes": ["ready evidence"],
                "verified": True,
                "score": 0.97,
            },
            {
                # The service must reject even graph-provided metadata for a
                # video outside the database-derived ready scope.
                "video_id": not_ready_youtube_video_id,
                "start_ms": 8_000,
                "end_ms": 9_000,
                "url": (
                    "https://www.youtube.com/watch?"
                    f"v={not_ready_youtube_video_id}&t=8s"
                ),
                "quotes": ["must not be exposed"],
                "verified": True,
                "score": 0.99,
            }
        ],
    )
    _install_real_message_handler(transactional_session_factory, graph)

    response = transactional_client.post(
        f"/threads/{thread_id}/messages",
        headers={"Authorization": f"Bearer {create_access_token(owner_id)}"},
        json={"content": "What does the ready video say?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["content"] == "The ready video says this."
    assert body["assistant_message"]["status"] == "answered"
    assert body["assistant_message"]["grounded"] is True

    assert len(graph.calls) == 1
    assert [message.content for message in graph.calls[0].messages] == [
        "What does the ready video say?"
    ]
    assert graph.calls[0].video_ids == [ready_youtube_video_id]

    user_message_id = uuid.UUID(body["user_message_id"])
    assistant_message_id = uuid.UUID(body["assistant_message"]["id"])

    with transactional_session_factory() as session:
        user_message = session.get(Message, user_message_id)
        assistant_message = session.get(Message, assistant_message_id)
        citations = list(
            session.scalars(
                select(Citation)
                .where(Citation.message_id == assistant_message_id)
                .order_by(Citation.position)
            )
        )

    assert user_message is not None
    assert user_message.role == MessageRole.USER
    assert user_message.status == MessageStatus.USER_MESSAGE
    assert user_message.content == "What does the ready video say?"
    assert user_message.rewritten_query == "What does the ready video say?"

    assert assistant_message is not None
    assert assistant_message.role == MessageRole.ASSISTANT
    assert assistant_message.status == MessageStatus.ANSWERED
    assert assistant_message.content == "The ready video says this."
    assert assistant_message.grounded is True

    assert len(citations) == 1
    citation = citations[0]
    assert citation.video_id == ready_video_id
    assert citation.start_ms == 1_000
    assert citation.end_ms == 4_500
    assert citation.quote_text == "ready evidence"
    assert citation.verified is True
    assert citation.verification_score == 0.97
    assert citation.position == 1


def test_post_message_hides_another_users_thread_before_calling_graph(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex

    with transactional_session_factory() as session:
        with session.begin():
            owner = User(
                email=f"private-thread-owner-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            other_user = User(
                email=f"private-thread-other-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add_all([owner, other_user])
            session.flush()

            thread, _ = _create_thread_source(
                session,
                owner=owner,
                ready_youtube_video_id=suffix[:11],
                source_status=SourceStatus.READY,
            )
            other_user_id = other_user.id
            thread_id = thread.id

    graph = RecordingGraph(
        answer="This must not run.",
        query="This must not run.",
        outcome="answered",
        grounded=True,
    )
    _install_real_message_handler(transactional_session_factory, graph)

    response = transactional_client.post(
        f"/threads/{thread_id}/messages",
        headers={
            "Authorization": f"Bearer {create_access_token(other_user_id)}"
        },
        json={"content": "Can I read this private chat?"},
    )

    assert response.status_code == 404
    assert graph.calls == []


def test_post_message_rejects_a_thread_without_ready_video_scope(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex

    with transactional_session_factory() as session:
        with session.begin():
            owner = User(
                email=f"empty-thread-owner-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(owner)
            session.flush()

            # This represents an existing thread whose source later lost its
            # only usable video.  The graph must never receive an empty scope.
            thread, _ = _create_thread_source(
                session,
                owner=owner,
                ready_youtube_video_id=None,
                source_status=SourceStatus.FAILED,
            )
            owner_id = owner.id
            thread_id = thread.id

    graph = RecordingGraph(
        answer="This must not run.",
        query="This must not run.",
        outcome="answered",
        grounded=True,
    )
    _install_real_message_handler(transactional_session_factory, graph)

    response = transactional_client.post(
        f"/threads/{thread_id}/messages",
        headers={"Authorization": f"Bearer {create_access_token(owner_id)}"},
        json={"content": "What happened to my video?"},
    )

    assert response.status_code == 409
    assert graph.calls == []


def test_post_message_persists_a_structural_no_context_refusal(
    transactional_session_factory,
    transactional_client,
) -> None:
    suffix = uuid.uuid4().hex
    youtube_video_id = suffix[:11]
    refusal = "I could not find that topic in this video."

    with transactional_session_factory() as session:
        with session.begin():
            owner = User(
                email=f"refusal-thread-owner-{suffix}@example.invalid",
                password_hash="test-only-not-a-real-password-hash",
            )
            session.add(owner)
            session.flush()

            thread, _ = _create_thread_source(
                session,
                owner=owner,
                ready_youtube_video_id=youtube_video_id,
                source_status=SourceStatus.READY,
            )
            owner_id = owner.id
            thread_id = thread.id

    graph = RecordingGraph(
        answer=refusal,
        query="Do they discuss quantum tunnelling?",
        outcome="refused_no_context",
        grounded=None,
    )
    _install_real_message_handler(transactional_session_factory, graph)

    response = transactional_client.post(
        f"/threads/{thread_id}/messages",
        headers={"Authorization": f"Bearer {create_access_token(owner_id)}"},
        json={"content": "Do they discuss quantum tunnelling?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]["content"] == refusal
    assert body["assistant_message"]["status"] == "refused_no_context"
    assert body["assistant_message"]["grounded"] is None
    assert body["assistant_message"]["citations"] == []

    user_message_id = uuid.UUID(body["user_message_id"])
    assistant_message_id = uuid.UUID(body["assistant_message"]["id"])

    with transactional_session_factory() as session:
        user_message = session.get(Message, user_message_id)
        assistant_message = session.get(Message, assistant_message_id)
        persisted_citations = list(
            session.scalars(
                select(Citation).where(
                    Citation.message_id == assistant_message_id
                )
            )
        )

    assert user_message is not None
    assert user_message.rewritten_query == "Do they discuss quantum tunnelling?"
    assert assistant_message is not None
    assert assistant_message.status == MessageStatus.REFUSED_NO_CONTEXT
    assert assistant_message.content == refusal
    assert assistant_message.grounded is None
    assert persisted_citations == []
