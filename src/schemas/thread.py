from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CreateThreadRequest(BaseModel):
    """Create a separate chat scoped to one uploaded source."""

    model_config = ConfigDict(extra="forbid")

    source_id: UUID
    title: str = Field(min_length=1, max_length=255)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        title = value.strip()

        if not title:
            raise ValueError("title cannot be blank.")

        return title


class ThreadResponse(BaseModel):
    id: UUID
    source_id: UUID
    title: str
    created_at: datetime
