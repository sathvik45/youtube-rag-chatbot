"""Persisted, source-scoped RAG turns for an existing chat thread.

The service deliberately has three phases: save the user turn, invoke RAG
outside a transaction, then save the assistant turn and citations.  This keeps
database connections available while the slower network/model work happens.
"""

import asyncio
from time import perf_counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.orm import Session

from src.db.models.message import Message, MessageRole, MessageStatus
from src.db.models.video import Video
from src.db.repositories.messages import (
    get_recent_messages,
    save_assistant_message,
    save_citations,
    save_user_message,
    set_user_message_rewritten_query,
)
from src.db.repositories.sources import get_ready_videos_for_source
from src.db.repositories.threads import get_thread_for_user, touch_thread
from src.db.session import SessionLocal
from src.core.config import settings
from src.core.logging import get_logger
from src.graph.build_graph import get_graph
from src.rag.Citations import strip_answer


SessionFactory = Callable[[], Session]
log = get_logger(__name__)


class ThreadNotFoundError(LookupError):
    """Raised when the current user cannot access the requested thread."""


class ThreadNotReadyForChatError(ValueError):
    """Raised when a thread currently has no ready source video to retrieve."""


class ChatProviderError(RuntimeError):
    """Raised when RAG cannot safely produce a completed assistant response."""


class GraphLike(Protocol):
    async def ainvoke(
        self,
        state: dict[str, Any],
        *,
        config: dict[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ChatCitation:
    id: UUID
    video_id: UUID
    start_ms: int
    end_ms: int
    youtube_url: str
    quote_text: str | None
    verified: bool | None
    verification_score: float | None
    position: int


@dataclass(frozen=True)
class ChatAssistantMessage:
    id: UUID
    content: str
    status: MessageStatus
    grounded: bool | None
    citations: tuple[ChatCitation, ...]


@dataclass(frozen=True)
class ChatTurn:
    user_message_id: UUID
    assistant_message: ChatAssistantMessage


ThreadMessageHandler = Callable[[UUID, UUID, str], Awaitable[ChatTurn]]


def _to_graph_messages(messages: list[Message]) -> list[BaseMessage]:
    """Translate durable chat history into the message types LangGraph uses."""
    graph_messages: list[BaseMessage] = []

    for message in messages:
        if message.role is MessageRole.USER:
            graph_messages.append(HumanMessage(content=message.content))
        elif message.role is MessageRole.ASSISTANT:
            graph_messages.append(AIMessage(content=message.content))

    return graph_messages


def _citation_rows(
    graph_citations: object,
    *,
    scoped_videos: dict[str, Video],
) -> list[dict[str, object]]:
    """Map graph citations back to database ids without escaping thread scope."""
    if not isinstance(graph_citations, list):
        return []

    rows: list[dict[str, object]] = []
    for citation in graph_citations:
        if not isinstance(citation, dict):
            continue

        youtube_video_id = citation.get("video_id")
        if not isinstance(youtube_video_id, str):
            continue

        video = scoped_videos.get(youtube_video_id)
        if video is None:
            # Citation metadata comes from retrieval/model output, so it is
            # never trusted to enlarge the source scope derived from the DB.
            continue

        try:
            start_ms = int(citation["start_ms"])
            end_ms = int(citation["end_ms"])
        except (KeyError, TypeError, ValueError):
            continue

        if start_ms < 0 or end_ms < start_ms:
            continue

        raw_score = citation.get("score")
        try:
            score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            score = None

        if score is not None and not 0 <= score <= 1:
            score = None

        verified = citation.get("verified")
        if not isinstance(verified, bool):
            verified = None

        raw_quotes = citation.get("quotes")
        if raw_quotes is None:
            raw_quotes = [citation.get("quote")]
        elif isinstance(raw_quotes, str):
            raw_quotes = [raw_quotes]

        quotes = (
            [quote for quote in raw_quotes if isinstance(quote, str) and quote]
            if isinstance(raw_quotes, list)
            else []
        )
        if not quotes:
            quotes = [None]

        youtube_url = citation.get("url")
        if not isinstance(youtube_url, str) or not youtube_url:
            youtube_url = video.canonical_url

        for quote in quotes:
            rows.append(
                {
                    "video_id": video.id,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "youtube_url": youtube_url,
                    "quote_text": quote,
                    "verified": verified,
                    "verification_score": score,
                }
            )

    return rows


def _citation_view(citation: Any) -> ChatCitation:
    return ChatCitation(
        id=citation.id,
        video_id=citation.video_id,
        start_ms=citation.start_ms,
        end_ms=citation.end_ms,
        youtube_url=citation.youtube_url,
        quote_text=citation.quote_text,
        verified=citation.verified,
        verification_score=citation.verification_score,
        position=citation.position,
    )


def _persist_temporary_error(
    *,
    user_id: UUID,
    thread_id: UUID,
    session_factory: SessionFactory,
) -> None:
    """Record a retryable assistant failure without exposing provider details."""
    session = session_factory()
    try:
        with session.begin():
            thread = get_thread_for_user(
                session,
                thread_id,
                user_id,
                for_update=True,
            )
            if thread is None:
                return

            save_assistant_message(
                session,
                thread_id=thread.id,
                content=(
                    "I couldn't complete that answer right now. "
                    "Please try again."
                ),
                grounded=None,
                status=MessageStatus.TEMPORARY_ERROR,
            )
            touch_thread(session, thread_id=thread.id)
    finally:
        session.close()


async def answer_thread_message(
    user_id: UUID,
    thread_id: UUID,
    content: str,
    *,
    session_factory: SessionFactory = SessionLocal,
    graph: GraphLike | None = None,
    history_limit: int = 20,
    rag_timeout_seconds: float | None = None,
) -> ChatTurn:
    """Answer one user turn using only videos linked to its owned thread.

    The graph call is intentionally made after the first transaction is
    committed and before the persistence transaction begins.  A slow LLM or
    vector store therefore cannot hold a PostgreSQL connection or lock open,
    and the request-level deadline bounds how long the caller waits.
    """
    content = content.strip()
    if not content:
        raise ValueError("content cannot be blank.")

    timeout_seconds = (
        settings.chat_rag_timeout_seconds
        if rag_timeout_seconds is None
        else rag_timeout_seconds
    )
    if timeout_seconds <= 0:
        raise ValueError("rag_timeout_seconds must be positive.")

    # Phase 1: authorize, derive scope from source links, and save the user
    # message before invoking any external RAG dependency.
    session = session_factory()
    try:
        with session.begin():
            thread = get_thread_for_user(
                session,
                thread_id,
                user_id,
                for_update=True,
            )
            if thread is None:
                raise ThreadNotFoundError(f"Thread {thread_id} was not found.")

            ready_videos = get_ready_videos_for_source(
                session,
                source_id=thread.source_id,
            )
            if not ready_videos:
                raise ThreadNotReadyForChatError(
                    "Thread source has no ready videos to use for chat."
                )

            user_message = save_user_message(
                session,
                thread_id=thread.id,
                content=content,
            )
            # Message inserts do not trigger Thread.updated_at's ORM onupdate
            # hook, so mark sidebar activity explicitly in this transaction.
            touch_thread(session, thread_id=thread.id)
            history = get_recent_messages(
                session,
                thread_id=thread.id,
                limit=history_limit,
            )

            user_message_id = user_message.id
            scoped_videos = {
                video.youtube_video_id: video for video in ready_videos
            }
            graph_messages = _to_graph_messages(history)
    finally:
        session.close()

    rag_started_at = perf_counter()
    try:
        runnable = graph if graph is not None else get_graph()
        graph_state = await asyncio.wait_for(
            runnable.ainvoke(
                {"messages": graph_messages},
                config={"configurable": {"video_ids": list(scoped_videos)}},
            ),
            timeout=timeout_seconds,
        )
        outcome = graph_state.get("outcome")
        if outcome not in {"answered", "refused_no_context"}:
            raise ValueError("Graph returned an invalid outcome.")

        output_messages = graph_state.get("messages")
        if not isinstance(output_messages, list) or not output_messages:
            raise ValueError("Graph returned no assistant message.")

        raw_answer = output_messages[-1].content
        if not isinstance(raw_answer, str):
            raise ValueError("Graph assistant message was not text.")

        answer = strip_answer(raw_answer)
        if not answer:
            raise ValueError("Graph assistant message was blank.")
    except Exception as error:
        rag_duration_ms = int((perf_counter() - rag_started_at) * 1000)
        try:
            _persist_temporary_error(
                user_id=user_id,
                thread_id=thread_id,
                session_factory=session_factory,
            )
        except Exception:
            # The route still returns a safe retryable error if the database is
            # unavailable while trying to record the provider failure.
            pass
        failure_type = (
            "timeout" if isinstance(error, TimeoutError) else type(error).__name__
        )
        log.warning(
            "chat_turn_failed thread_id=%s user_message_id=%s "
            "failure_type=%s rag_duration_ms=%s timeout_seconds=%s",
            thread_id,
            user_message_id,
            failure_type,
            rag_duration_ms,
            timeout_seconds,
        )
        raise ChatProviderError("Chat is temporarily unavailable.") from error

    rag_duration_ms = int((perf_counter() - rag_started_at) * 1000)

    message_status = (
        MessageStatus.ANSWERED
        if outcome == "answered"
        else MessageStatus.REFUSED_NO_CONTEXT
    )
    grounded = graph_state.get("grounded")
    if not isinstance(grounded, bool):
        grounded = None
    citation_rows = (
        _citation_rows(
            graph_state.get("citations"),
            scoped_videos=scoped_videos,
        )
        if outcome == "answered"
        else []
    )
    rewritten_query = graph_state.get("query")
    if not isinstance(rewritten_query, str):
        rewritten_query = None

    # Phase 3: atomically save the completed assistant answer and all of its
    # citations.  A partially persisted answer can never be returned.
    session = session_factory()
    try:
        with session.begin():
            thread = get_thread_for_user(
                session,
                thread_id,
                user_id,
                for_update=True,
            )
            if thread is None:
                raise ThreadNotFoundError(f"Thread {thread_id} was not found.")

            set_user_message_rewritten_query(
                session,
                message_id=user_message_id,
                rewritten_query=rewritten_query,
            )
            assistant_message = save_assistant_message(
                session,
                thread_id=thread.id,
                content=answer,
                grounded=grounded,
                status=message_status,
            )
            citations = save_citations(
                session,
                message_id=assistant_message.id,
                citations=citation_rows,
            )
            touch_thread(session, thread_id=thread.id)

            completed_turn = ChatTurn(
                user_message_id=user_message_id,
                assistant_message=ChatAssistantMessage(
                    id=assistant_message.id,
                    content=assistant_message.content,
                    status=assistant_message.status,
                    grounded=assistant_message.grounded,
                    citations=tuple(_citation_view(citation) for citation in citations),
                ),
            )
    finally:
        session.close()

    # This runs after the transaction has committed, so the completion event
    # never claims success for an assistant turn that failed to persist.
    log.info(
        "chat_turn_completed thread_id=%s user_message_id=%s "
        "assistant_message_id=%s outcome=%s grounded=%s "
        "citation_count=%s rag_duration_ms=%s",
        thread_id,
        user_message_id,
        completed_turn.assistant_message.id,
        outcome,
        grounded,
        len(completed_turn.assistant_message.citations),
        rag_duration_ms,
    )
    return completed_turn
