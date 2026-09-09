"""change document_chunks embedding to vector(512)

Revision ID: 68e1e41ad617
Revises: 96391ea59e58
Create Date: 2026-09-08 13:50:58.984093

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '68e1e41ad617'
down_revision: Union[str, Sequence[str], None] = '96391ea59e58'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(512)")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(384)")
