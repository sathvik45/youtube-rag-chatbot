from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class CreateSourceRequest(BaseModel):
    """Temporary development request; authentication will replace user_id."""

    user_id: UUID
    youtube_url: str = Field(
        min_length=1,
        max_length=2048,
    )

    @field_validator("youtube_url")
    @classmethod
    def validate_youtube_url(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("youtube_url cannot be blank.")

        return value


class SourceSubmissionResponse(BaseModel):
    source_id: UUID
    video_ids: list[UUID]
    job_ids: list[UUID]


class SourceProgressResponse(BaseModel):
    source_id: UUID
    status: str
    video_counts: dict[str, int]
    job_counts: dict[str, int]
    total_videos: int
    completed_videos: int