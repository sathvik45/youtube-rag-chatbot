from collections.abc import Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.db.repositories.sources import SourceProgress
from src.schemas.source import (
    CreateSourceRequest,
    SourceProgressResponse,
    SourceSubmissionResponse,
    SourceListItem,
    SourceListResponse,
)
from src.services.source_submission import SourceSubmission

from src.api.deps import (
    SourceProgressReader,
    SourceSubmitter,
    get_current_user,
    get_source_progress_reader,
    get_source_submitter,
    SourceListReader,
    get_source_list_reader,

)
from src.services.auth import AuthenticatedUser
from src.api.cursors import decode_source_cursor, encode_source_cursor

router = APIRouter(
    prefix="/sources",
    tags=["sources"],
)


@router.post("", response_model=SourceSubmissionResponse, status_code=201)
def create_source(
    request: CreateSourceRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    submitter: SourceSubmitter = Depends(get_source_submitter),
) -> SourceSubmissionResponse:
    try:
        submission: SourceSubmission = submitter(
            current_user.id,
            request.youtube_url,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    return SourceSubmissionResponse(
        source_id=submission.source_id,
        video_ids=submission.video_ids,
        job_ids=submission.job_ids,
    )

@router.get(
    "/{source_id}",
    response_model=SourceProgressResponse,
)
def get_source_status(
    source_id: UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    progress_reader: SourceProgressReader = Depends(
        get_source_progress_reader
    ),
) -> SourceProgressResponse:
    """Return ingestion progress for one source owned by the current user."""
    try:
        progress: SourceProgress = progress_reader(
            current_user.id,
            source_id,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source not found.",
        ) from error

    return SourceProgressResponse(
        source_id=progress.source_id,
        status=progress.source_status.value,
        video_counts=progress.video_counts,
        job_counts=progress.job_counts,
        total_videos=progress.total_videos,
        completed_videos=progress.completed_videos,
    )

@router.get("", response_model=SourceListResponse)
def list_sources(
    current_user: AuthenticatedUser = Depends(get_current_user),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    source_list_reader: SourceListReader = Depends(
        get_source_list_reader
    ),
) -> SourceListResponse:
    """Return one newest-first page of the current user's sources."""
    before_created_at = None
    before_id = None

    if cursor is not None:
        try:
            before_created_at, before_id = decode_source_cursor(cursor)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Invalid source cursor.",
            ) from error

    sources = source_list_reader(
        current_user.id,
        before_created_at,
        before_id,
        limit + 1,
    )

    visible_sources = sources[:limit]
    next_cursor = None

    if len(sources) > limit and visible_sources:
        last_source = visible_sources[-1]
        next_cursor = encode_source_cursor(
            created_at=last_source.created_at,
            source_id=last_source.id,
        )

    return SourceListResponse(
        items=[
            SourceListItem(
                id=source.id,
                submitted_value=source.submitted_value,
                source_type=source.source_type.value,
                status=source.status.value,
                created_at=source.created_at,
            )
            for source in visible_sources
        ],
        next_cursor=next_cursor,
    )