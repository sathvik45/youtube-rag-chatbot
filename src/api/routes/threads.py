from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import (
    ThreadCreator,
    ThreadMessageHandler,
    get_current_user,
    get_thread_creator,
    get_thread_message_handler,
)
from src.schemas.message import (
    AssistantMessageResponse,
    CitationResponse,
    CreateMessageRequest,
    ThreadMessageResponse,
)
from src.schemas.thread import CreateThreadRequest, ThreadResponse
from src.services.auth import AuthenticatedUser
from src.services.threads import (
    CreatedThread,
    SourceNotReadyForChatError,
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
                CitationResponse(
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
                for citation in turn.assistant_message.citations
            ],
        ),
    )
