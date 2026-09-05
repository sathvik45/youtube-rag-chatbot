"""remove thread archive state

Revision ID: a4f1c2b3d4e5
Revises: dfc20681eec1
Create Date: 2026-09-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a4f1c2b3d4e5"
down_revision: Union[str, Sequence[str], None] = "dfc20681eec1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove the unused archive timestamp from chat threads."""
    op.drop_column("threads", "archived_at")


def downgrade() -> None:
    """Restore the archive timestamp for a downgrade."""
    op.add_column(
        "threads",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )
