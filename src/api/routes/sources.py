from collections.abc import Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import (
    SourceProgressReader,
    SourceSubmitter,
    get_source_progress_reader,
    get_source_submitter,
)
from src.db.repositories.sources import SourceProgress
from src.schemas.source import (
    CreateSourceRequest,
    SourceProgressResponse,
    SourceSubmissionResponse,
)
from src.services.source_submission import SourceSubmission


router = APIRouter(
    prefix="/sources",
    tags=["sources"],
)


@router.post(
    "",
    response_model=SourceSubmissionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_source(
    request: CreateSourceRequest,
    submitter: SourceSubmitter = Depends(get_source_submitter),
) -> SourceSubmissionResponse:
    """Submit a YouTube video or playlist and queue required ingestion jobs."""
    try:
        submission: SourceSubmission = submitter(
            request.user_id,
            request.youtube_url,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    return SourceSubmissionResponse(
        source_id=submission.source_id,
        video_ids=list(submission.video_ids),
        job_ids=list(submission.job_ids),
    )


@router.get(
    "/{source_id}",
    response_model=SourceProgressResponse,
)
def get_source_status(
    source_id: UUID,
    progress_reader: SourceProgressReader = Depends(
        get_source_progress_reader
    ),
) -> SourceProgressResponse:
    """Return ingestion progress for one source."""
    try:
        progress: SourceProgress = progress_reader(source_id)
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    return SourceProgressResponse(
        source_id=progress.source_id,
        status=progress.source_status.value,
        video_counts=progress.video_counts,
        job_counts=progress.job_counts,
        total_videos=progress.total_videos,
        completed_videos=progress.completed_videos,
    )