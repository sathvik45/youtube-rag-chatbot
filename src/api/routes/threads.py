from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import ThreadCreator, get_current_user, get_thread_creator
from src.schemas.thread import CreateThreadRequest, ThreadResponse
from src.services.auth import AuthenticatedUser
from src.services.threads import (
    CreatedThread,
    SourceNotReadyForChatError,
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
