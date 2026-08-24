"""create core chat tables

Revision ID: 2e49fd95ac85
Revises: 
Create Date: 2026-08-24 18:11:22.453603

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '2e49fd95ac85'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the initial persistence schema for chats and ingestion."""
    source_type = postgresql.ENUM("video", "playlist", name="source_type", create_type=False)
    source_status = postgresql.ENUM(
        "pending", "processing", "ready", "partial", "failed",
        name="source_status", create_type=False,
    )
    transcript_status = postgresql.ENUM(
        "pending", "processing", "ready", "no_transcript", "failed",
        name="transcript_status", create_type=False,
    )
    source_video_status = postgresql.ENUM(
        "pending", "processing", "ready", "no_transcript", "failed",
        name="source_video_status", create_type=False,
    )
    message_role = postgresql.ENUM(
        "user", "assistant", "system", name="message_role", create_type=False
    )
    message_status = postgresql.ENUM(
        "user_message", "answered", "refused_no_context",
        "refused_insufficient_context", "temporary_error", "failed",
        name="message_status", create_type=False,
    )
    ingestion_job_status = postgresql.ENUM(
        "queued", "running", "succeeded", "failed", "retrying",
        name="ingestion_job_status", create_type=False,
    )

    bind = op.get_bind()
    for enum_type in (
        source_type, source_status, transcript_status, source_video_status,
        message_role, message_status, ingestion_job_status,
    ):
        enum_type.create(bind, checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "videos",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("youtube_video_id", sa.String(length=11), nullable=False),
        sa.Column("canonical_url", sa.String(length=2048), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("transcript_status", transcript_status, nullable=False),
        sa.Column("transcript_language", sa.String(length=20), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("vector_count", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("chunk_count >= 0", name="ck_videos_chunk_count_nonnegative"),
        sa.CheckConstraint("vector_count >= 0", name="ck_videos_vector_count_nonnegative"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_videos_transcript_status", "videos", ["transcript_status"])
    op.create_index("ix_videos_youtube_video_id", "videos", ["youtube_video_id"], unique=True)

    op.create_table(
        "sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_type", source_type, nullable=False),
        sa.Column("submitted_value", sa.String(length=2048), nullable=False),
        sa.Column("youtube_playlist_id", sa.String(length=100), nullable=True),
        sa.Column("status", source_status, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sources_status", "sources", ["status"])
    op.create_index("ix_sources_user_id", "sources", ["user_id"])
    op.create_index("ix_sources_youtube_playlist_id", "sources", ["youtube_playlist_id"])

    op.create_table(
        "source_videos",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("status", source_video_status, nullable=False),
        sa.Column("error_message", sa.String(length=2000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("position >= 0", name="ck_source_videos_position_nonnegative"),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id", "video_id", name="uq_source_video"),
        sa.UniqueConstraint("source_id", "position", name="uq_source_video_position"),
    )
    op.create_index("ix_source_videos_source_id", "source_videos", ["source_id"])
    op.create_index("ix_source_videos_status", "source_videos", ["status"])
    op.create_index("ix_source_videos_video_id", "source_videos", ["video_id"])

    op.create_table(
        "threads",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_threads_source_id", "threads", ["source_id"])
    op.create_index("ix_threads_user_id", "threads", ["user_id"])

    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("thread_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", message_role, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("rewritten_query", sa.Text(), nullable=True),
        sa.Column("grounded", sa.Boolean(), nullable=True),
        sa.Column("status", message_status, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["thread_id"], ["threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_created_at", "messages", ["created_at"])
    op.create_index("ix_messages_thread_id", "messages", ["thread_id"])

    op.create_table(
        "citations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("youtube_url", sa.Text(), nullable=False),
        sa.Column("quote_text", sa.Text(), nullable=True),
        sa.Column("verified", sa.Boolean(), nullable=True),
        sa.Column("verification_score", sa.Float(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.CheckConstraint("start_ms >= 0", name="ck_citations_start_ms_nonnegative"),
        sa.CheckConstraint("end_ms >= start_ms", name="ck_citations_time_range"),
        sa.CheckConstraint(
            "verification_score >= 0 AND verification_score <= 1",
            name="ck_citations_verification_score",
        ),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("message_id", "position", name="uq_citations_message_position"),
    )
    op.create_index("ix_citations_message_id", "citations", ["message_id"])
    op.create_index("ix_citations_video_id", "citations", ["video_id"])

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("status", ingestion_job_status, nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ingestion_jobs_created_at", "ingestion_jobs", ["created_at"])
    op.create_index("ix_ingestion_jobs_source_id", "ingestion_jobs", ["source_id"])
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status"])
    op.create_index("ix_ingestion_jobs_video_id", "ingestion_jobs", ["video_id"])


def downgrade() -> None:
    """Remove the core chat schema in dependency-safe order."""
    op.drop_table("ingestion_jobs")
    op.drop_table("citations")
    op.drop_table("messages")
    op.drop_table("threads")
    op.drop_table("source_videos")
    op.drop_table("sources")
    op.drop_table("videos")
    op.drop_table("users")

    bind = op.get_bind()
    for enum_type in (
        postgresql.ENUM(name="ingestion_job_status"),
        postgresql.ENUM(name="message_status"),
        postgresql.ENUM(name="message_role"),
        postgresql.ENUM(name="source_video_status"),
        postgresql.ENUM(name="transcript_status"),
        postgresql.ENUM(name="source_status"),
        postgresql.ENUM(name="source_type"),
    ):
        enum_type.drop(bind, checkfirst=True)
