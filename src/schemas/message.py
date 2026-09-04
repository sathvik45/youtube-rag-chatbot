from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CreateMessageRequest(BaseModel):
    """A single user turn sent to an existing source-scoped thread."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        content = value.strip()

        if not content:
            raise ValueError("content cannot be blank.")

        return content


class CitationResponse(BaseModel):
    id: UUID
    video_id: UUID
    start_ms: int
    end_ms: int
    youtube_url: str
    quote_text: str | None
    verified: bool | None
    verification_score: float | None
    position: int


class AssistantMessageResponse(BaseModel):
    id: UUID
    content: str
    status: str
    grounded: bool | None
    citations: list[CitationResponse]


class ThreadMessageResponse(BaseModel):
    user_message_id: UUID
    assistant_message: AssistantMessageResponse
