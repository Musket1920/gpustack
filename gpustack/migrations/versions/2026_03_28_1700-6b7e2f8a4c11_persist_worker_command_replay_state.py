"""persist worker command replay state

Revision ID: 6b7e2f8a4c11
Revises: bf4c3d9e21a1
Create Date: 2026-03-28 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from gpustack.migrations.utils import column_exists, table_exists


# revision identifiers, used by Alembic.
revision: str = '6b7e2f8a4c11'
down_revision: Union[str, None] = 'bf4c3d9e21a1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def index_exists(table_name: str, index_name: str) -> bool:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    indexes = [index["name"] for index in inspector.get_indexes(table_name)]
    return index_name in indexes


def unique_constraint_exists(table_name: str, constraint_name: str) -> bool:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    constraints = [
        constraint["name"]
        for constraint in inspector.get_unique_constraints(table_name)
        if constraint.get("name") is not None
    ]
    return constraint_name in constraints


def upgrade() -> None:
    if table_exists('worker_sessions'):
        with op.batch_alter_table('worker_sessions', schema=None) as batch_op:
            if not column_exists('worker_sessions', 'generation'):
                batch_op.add_column(
                    sa.Column('generation', sa.Integer(), nullable=False, server_default='1')
                )
            if not column_exists('worker_sessions', 'last_command_sequence'):
                batch_op.add_column(
                    sa.Column(
                        'last_command_sequence',
                        sa.Integer(),
                        nullable=False,
                        server_default='0',
                    )
                )
            if not column_exists('worker_sessions', 'last_acknowledged_command_sequence'):
                batch_op.add_column(
                    sa.Column(
                        'last_acknowledged_command_sequence',
                        sa.Integer(),
                        nullable=False,
                        server_default='0',
                    )
                )
            if not column_exists('worker_sessions', 'last_completed_command_sequence'):
                batch_op.add_column(
                    sa.Column(
                        'last_completed_command_sequence',
                        sa.Integer(),
                        nullable=False,
                        server_default='0',
                    )
                )
            if not column_exists('worker_sessions', 'replay_cursor'):
                batch_op.add_column(
                    sa.Column(
                        'replay_cursor',
                        sa.Integer(),
                        nullable=False,
                        server_default='0',
                    )
                )
            if not column_exists('worker_sessions', 'requires_full_reconcile'):
                batch_op.add_column(
                    sa.Column(
                        'requires_full_reconcile',
                        sa.Boolean(),
                        nullable=False,
                        server_default=sa.false(),
                    )
                )
            if not column_exists('worker_sessions', 'full_reconcile_reason'):
                batch_op.add_column(
                    sa.Column('full_reconcile_reason', sa.Text(), nullable=True)
                )

        if not index_exists(
            'worker_sessions', 'ix_worker_sessions_worker_id_generation'
        ):
            op.create_index(
                'ix_worker_sessions_worker_id_generation',
                'worker_sessions',
                ['worker_id', 'generation'],
                unique=False,
            )
        if not unique_constraint_exists(
            'worker_sessions', 'uq_worker_sessions_worker_id_generation'
        ):
            op.create_unique_constraint(
                'uq_worker_sessions_worker_id_generation',
                'worker_sessions',
                ['worker_id', 'generation'],
            )

    if table_exists('worker_commands'):
        with op.batch_alter_table('worker_commands', schema=None) as batch_op:
            if not column_exists('worker_commands', 'sequence'):
                batch_op.add_column(
                    sa.Column('sequence', sa.Integer(), nullable=False, server_default='0')
                )
            if not column_exists('worker_commands', 'worker_session_generation'):
                batch_op.add_column(
                    sa.Column('worker_session_generation', sa.Integer(), nullable=True)
                )

        if not index_exists(
            'worker_commands', 'ix_worker_commands_worker_id_sequence'
        ):
            op.create_index(
                'ix_worker_commands_worker_id_sequence',
                'worker_commands',
                ['worker_id', 'sequence'],
                unique=False,
            )
        if not index_exists(
            'worker_commands', 'ix_worker_commands_worker_id_idempotency_key'
        ):
            op.create_index(
                'ix_worker_commands_worker_id_idempotency_key',
                'worker_commands',
                ['worker_id', 'idempotency_key'],
                unique=True,
            )
        if not unique_constraint_exists(
            'worker_commands', 'uq_worker_commands_worker_id_sequence'
        ):
            op.create_unique_constraint(
                'uq_worker_commands_worker_id_sequence',
                'worker_commands',
                ['worker_id', 'sequence'],
            )


def downgrade() -> None:
    if table_exists('worker_commands'):
        if unique_constraint_exists(
            'worker_commands', 'uq_worker_commands_worker_id_sequence'
        ):
            op.drop_constraint(
                'uq_worker_commands_worker_id_sequence',
                'worker_commands',
                type_='unique',
            )
        if index_exists('worker_commands', 'ix_worker_commands_worker_id_idempotency_key'):
            op.drop_index(
                'ix_worker_commands_worker_id_idempotency_key',
                table_name='worker_commands',
            )
        if index_exists('worker_commands', 'ix_worker_commands_worker_id_sequence'):
            op.drop_index(
                'ix_worker_commands_worker_id_sequence',
                table_name='worker_commands',
            )

        with op.batch_alter_table('worker_commands', schema=None) as batch_op:
            if column_exists('worker_commands', 'worker_session_generation'):
                batch_op.drop_column('worker_session_generation')
            if column_exists('worker_commands', 'sequence'):
                batch_op.drop_column('sequence')

    if table_exists('worker_sessions'):
        if unique_constraint_exists(
            'worker_sessions', 'uq_worker_sessions_worker_id_generation'
        ):
            op.drop_constraint(
                'uq_worker_sessions_worker_id_generation',
                'worker_sessions',
                type_='unique',
            )
        if index_exists('worker_sessions', 'ix_worker_sessions_worker_id_generation'):
            op.drop_index(
                'ix_worker_sessions_worker_id_generation',
                table_name='worker_sessions',
            )

        with op.batch_alter_table('worker_sessions', schema=None) as batch_op:
            if column_exists('worker_sessions', 'full_reconcile_reason'):
                batch_op.drop_column('full_reconcile_reason')
            if column_exists('worker_sessions', 'requires_full_reconcile'):
                batch_op.drop_column('requires_full_reconcile')
            if column_exists('worker_sessions', 'replay_cursor'):
                batch_op.drop_column('replay_cursor')
            if column_exists('worker_sessions', 'last_completed_command_sequence'):
                batch_op.drop_column('last_completed_command_sequence')
            if column_exists('worker_sessions', 'last_acknowledged_command_sequence'):
                batch_op.drop_column('last_acknowledged_command_sequence')
            if column_exists('worker_sessions', 'last_command_sequence'):
                batch_op.drop_column('last_command_sequence')
            if column_exists('worker_sessions', 'generation'):
                batch_op.drop_column('generation')
