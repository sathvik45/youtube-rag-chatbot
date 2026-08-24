import uuid
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageStatus(str, Enum):
    USER_MESSAGE = "user_message"
    ANSWERED = "answered"
    REFUSED_NO_CONTEXT = "refused_no_context"
    REFUSED_INSUFFICIENT_CONTEXT = "refused_insufficient_context"
    TEMPORARY_ERROR = "temporary_error"
    FAILED = "failed"


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    thread_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("threads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    role: Mapped[MessageRole] = mapped_column(
        SQLEnum(
            MessageRole,
            name="message_role",
            native_enum=True,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    rewritten_query: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    grounded: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    status: Mapped[MessageStatus] = mapped_column(
        SQLEnum(
            MessageStatus,
            name="message_status",
            native_enum=True,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=MessageStatus.USER_MESSAGE,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
