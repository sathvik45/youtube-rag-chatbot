from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from src.api.deps import (
    ThreadCreator,
    ThreadDeleter,
    ThreadHistoryReader,
    ThreadListReader,
    ThreadMessageHandler,
    get_current_user,
    get_thread_deleter,
    get_thread_creator,
    get_thread_history_reader,
    get_thread_list_reader,
    get_thread_message_handler,
)
from src.api.cursors import (
    decode_message_cursor,
    decode_thread_cursor,
    encode_message_cursor,
    encode_thread_cursor,
)
from src.schemas.message import (
    AssistantMessageResponse,
    CitationResponse,
    CreateMessageRequest,
    ThreadHistoryMessageResponse,
    ThreadHistoryResponse,
    ThreadMessageResponse,
)
from src.schemas.thread import (
    CreateThreadRequest,
    ThreadListItem,
    ThreadListResponse,
    ThreadResponse,
)
from src.services.auth import AuthenticatedUser
from src.services.threads import (
    CreatedThread,
    SourceNotReadyForChatError,
    ThreadHistory,
)
from src.services.chat import (
    ChatProviderError,
    ChatTurn,
    ThreadNotReadyForChatError,
)


router = APIRouter(
    prefix="/threads",
    tags=["threads"],
)


def _citation_response(citation) -> CitationResponse:
    return CitationResponse(
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


@router.post("", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
def create_thread(
    request: CreateThreadRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    thread_creator: ThreadCreator = Depends(get_thread_creator),
) -> ThreadResponse:
    """Create one authenticated user's chat scoped to one uploaded source."""
    try:
        created_thread: CreatedThread = thread_creator(
            current_user.id,
            request.source_id,
            request.title,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source not found.",
        ) from error
    except SourceNotReadyForChatError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error

    return ThreadResponse(
        id=created_thread.id,
        source_id=created_thread.source_id,
        title=created_thread.title,
        created_at=created_thread.created_at,
    )


@router.get("", response_model=ThreadListResponse)
def list_threads(
    current_user: AuthenticatedUser = Depends(get_current_user),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    thread_list_reader: ThreadListReader = Depends(get_thread_list_reader),
) -> ThreadListResponse:
    """Return one newest page from the authenticated user's chat sidebar."""
    before_updated_at = None
    before_id = None
    if cursor is not None:
        try:
            before_updated_at, before_id = decode_thread_cursor(cursor)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Invalid thread cursor.",
            ) from error

    threads = thread_list_reader(
        current_user.id,
        before_updated_at,
        before_id,
        limit + 1,
    )
    visible_threads = threads[:limit]
    next_cursor = None
    if len(threads) > limit and visible_threads:
        last_thread = visible_threads[-1]
        next_cursor = encode_thread_cursor(
            updated_at=last_thread.updated_at,
            thread_id=last_thread.id,
        )

    return ThreadListResponse(
        items=[
            ThreadListItem(
                id=thread.id,
                source_id=thread.source_id,
                title=thread.title,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
            )
            for thread in visible_threads
        ],
        next_cursor=next_cursor,
    )


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_thread(
    thread_id: UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    thread_deleter: ThreadDeleter = Depends(
        get_thread_deleter
    ),
) -> Response:
    """Permanently delete an owned chat and its saved messages/citations."""
    try:
        thread_deleter(
            current_user.id,
            thread_id,
        )
    except LookupError as error:
        # Ownership failures intentionally look identical to missing threads.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found.",
        ) from error

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{thread_id}/messages", response_model=ThreadMessageResponse)
async def create_thread_message(
    thread_id: UUID,
    request: CreateMessageRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    message_handler: ThreadMessageHandler = Depends(
        get_thread_message_handler
    ),
) -> ThreadMessageResponse:
    """Save one user turn and return a RAG answer scoped to this thread."""
    try:
        turn: ChatTurn = await message_handler(
            current_user.id,
            thread_id,
            request.content,
        )
    except LookupError as error:
        # Ownership failures intentionally look identical to missing threads.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found.",
        ) from error
    except ThreadNotReadyForChatError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except ChatProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat is temporarily unavailable. Please try again.",
        ) from error

    return ThreadMessageResponse(
        user_message_id=turn.user_message_id,
        assistant_message=AssistantMessageResponse(
            id=turn.assistant_message.id,
            content=turn.assistant_message.content,
            status=turn.assistant_message.status.value,
            grounded=turn.assistant_message.grounded,
            citations=[
                _citation_response(citation)
                for citation in turn.assistant_message.citations
            ],
        ),
    )


@router.get("/{thread_id}/messages", response_model=ThreadHistoryResponse)
def get_thread_messages(
    thread_id: UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    thread_history_reader: ThreadHistoryReader = Depends(
        get_thread_history_reader
    ),
) -> ThreadHistoryResponse:
    """Restore a source-scoped chat page in chronological display order."""
    before_created_at = None
    before_id = None
    if cursor is not None:
        try:
            before_created_at, before_id = decode_message_cursor(cursor)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Invalid message cursor.",
            ) from error

    try:
        history: ThreadHistory = thread_history_reader(
            current_user.id,
            thread_id,
            before_created_at,
            before_id,
            limit + 1,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found.",
        ) from error

    newest_first = history.messages[:limit]
    next_cursor = None
    if len(history.messages) > limit and newest_first:
        oldest_visible = newest_first[-1]
        next_cursor = encode_message_cursor(
            created_at=oldest_visible.created_at,
            message_id=oldest_visible.id,
        )

    return ThreadHistoryResponse(
        thread_id=history.thread_id,
        items=[
            ThreadHistoryMessageResponse(
                id=message.id,
                role=message.role.value,
                content=message.content,
                rewritten_query=message.rewritten_query,
                grounded=message.grounded,
                status=message.status.value,
                created_at=message.created_at,
                citations=[
                    _citation_response(citation)
                    for citation in message.citations
                ],
            )
            for message in reversed(newest_first)
        ],
        next_cursor=next_cursor,
    )
