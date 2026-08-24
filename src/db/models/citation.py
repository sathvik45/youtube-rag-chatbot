import uuid
from uuid import UUID

from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class Citation(Base):
    __tablename__ = "citations"
    __table_args__ = (
        CheckConstraint("start_ms >= 0", name="ck_citations_start_ms_nonnegative"),
        CheckConstraint("end_ms >= start_ms", name="ck_citations_time_range"),
        CheckConstraint(
            "verification_score >= 0 AND verification_score <= 1",
            name="ck_citations_verification_score",
        ),
        UniqueConstraint("message_id", "position", name="uq_citations_message_position"),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    message_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    video_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    start_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    end_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    youtube_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    quote_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    verified: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    verification_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    position: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
