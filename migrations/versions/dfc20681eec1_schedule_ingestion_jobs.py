"""schedule_ingestion_jobs

Revision ID: dfc20681eec1
Revises: 2e49fd95ac85
Create Date: 2026-09-02 22:28:45.468845

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dfc20681eec1'
down_revision: Union[str, Sequence[str], None] = '2e49fd95ac85'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    op.execute(
        """
        UPDATE ingestion_jobs
        SET available_at = COALESCE(finished_at, created_at)
        WHERE available_at IS NULL
        """
    )

    op.alter_column(
        "ingestion_jobs",
        "available_at",
        nullable=False,
    )

    op.add_column(
        "ingestion_jobs",
        sa.Column(
            "lease_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # Any job already marked running predates leases.
    # Make it recoverable when we add stale-job recovery next.
    op.execute(
        """
        UPDATE ingestion_jobs
        SET lease_expires_at = CURRENT_TIMESTAMP
        WHERE status = 'running'
        """
    )

    op.create_index(
        "ix_ingestion_jobs_runnable",
        "ingestion_jobs",
        ["status", "available_at", "created_at", "id"],
        unique=False,
    )

    op.create_index(
        "ix_ingestion_jobs_running_lease",
        "ingestion_jobs",
        ["status", "lease_expires_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ingestion_jobs_running_lease",
        table_name="ingestion_jobs",
    )
    op.drop_index(
        "ix_ingestion_jobs_runnable",
        table_name="ingestion_jobs",
    )
    op.drop_column("ingestion_jobs", "lease_expires_at")
    op.drop_column("ingestion_jobs", "available_at")