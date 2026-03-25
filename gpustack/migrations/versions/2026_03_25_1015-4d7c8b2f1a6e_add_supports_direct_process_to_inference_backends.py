"""add supports_direct_process to inference_backends

Revision ID: 4d7c8b2f1a6e
Revises: 8ad0f94c92e8
Create Date: 2026-03-25 10:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.migrations.utils import column_exists


# revision identifiers, used by Alembic.
revision: str = '4d7c8b2f1a6e'
down_revision: Union[str, None] = '8ad0f94c92e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        if not column_exists('inference_backends', 'supports_direct_process'):
            batch_op.add_column(
                sa.Column('supports_direct_process', sa.Boolean(), nullable=True)
            )


def downgrade() -> None:
    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        if column_exists('inference_backends', 'supports_direct_process'):
            batch_op.drop_column('supports_direct_process')
