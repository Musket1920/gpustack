"""add worker outbound control contracts

Revision ID: bf4c3d9e21a1
Revises: 4d7c8b2f1a6e
Create Date: 2026-03-28 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.migrations.utils import column_exists, table_exists
from gpustack.schemas.common import UTCDateTime


# revision identifiers, used by Alembic.
revision: str = 'bf4c3d9e21a1'
down_revision: Union[str, None] = '4d7c8b2f1a6e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('workers', schema=None) as batch_op:
        if not column_exists('workers', 'capabilities'):
            batch_op.add_column(sa.Column('capabilities', sa.JSON(), nullable=True))
        if not column_exists('workers', 'reachability_mode'):
            batch_op.add_column(
                sa.Column(
                    'reachability_mode',
                    sa.String(),
                    nullable=True,
                )
            )

    conn = op.get_bind()
    conn.execute(
        sa.text(
            "UPDATE workers SET reachability_mode = 'reverse_probe' WHERE reachability_mode IS NULL"
        )
    )

    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.alter_column(
            'reachability_mode',
            existing_type=sa.String(),
            nullable=False,
        )

    if not table_exists('worker_sessions'):
        op.create_table(
            'worker_sessions',
            sa.Column('created_at', UTCDateTime(), nullable=False),
            sa.Column('updated_at', UTCDateTime(), nullable=False),
            sa.Column('deleted_at', UTCDateTime(), nullable=True),
            sa.Column('session_id', sa.String(), nullable=False, unique=True),
            sa.Column('worker_id', sa.Integer(), sa.ForeignKey('workers.id', ondelete='CASCADE'), nullable=False),
            sa.Column('control_channel', sa.String(), nullable=False),
            sa.Column('reachability_mode', sa.String(), nullable=False),
            sa.Column('state', sa.String(), nullable=False, server_default='active'),
            sa.Column('protocol_version', sa.Integer(), nullable=False, server_default='1'),
            sa.Column('connected_at', UTCDateTime(), nullable=True),
            sa.Column('last_seen_at', UTCDateTime(), nullable=True),
            sa.Column('disconnected_at', UTCDateTime(), nullable=True),
            sa.Column('expires_at', UTCDateTime(), nullable=True),
            sa.Column('details', sa.JSON(), nullable=True),
            sa.Column('id', sa.Integer(), nullable=False),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index(
            op.f('ix_worker_sessions_session_id'),
            'worker_sessions',
            ['session_id'],
            unique=True,
        )
        op.create_index(
            op.f('ix_worker_sessions_worker_id'),
            'worker_sessions',
            ['worker_id'],
            unique=False,
        )

    if not table_exists('worker_commands'):
        op.create_table(
            'worker_commands',
            sa.Column('created_at', UTCDateTime(), nullable=False),
            sa.Column('updated_at', UTCDateTime(), nullable=False),
            sa.Column('deleted_at', UTCDateTime(), nullable=True),
            sa.Column('command_id', sa.String(), nullable=False, unique=True),
            sa.Column('worker_id', sa.Integer(), sa.ForeignKey('workers.id', ondelete='CASCADE'), nullable=False),
            sa.Column('worker_session_id', sa.Integer(), sa.ForeignKey('worker_sessions.id', ondelete='SET NULL'), nullable=True),
            sa.Column('command_type', sa.String(), nullable=False),
            sa.Column('payload', sa.JSON(), nullable=False),
            sa.Column('state', sa.String(), nullable=False, server_default='pending'),
            sa.Column('idempotency_key', sa.String(), nullable=True),
            sa.Column('dispatch_attempts', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('desired_worker_state_revision', sa.Integer(), nullable=True),
            sa.Column('lease_expires_at', UTCDateTime(), nullable=True),
            sa.Column('not_before', UTCDateTime(), nullable=True),
            sa.Column('dispatched_at', UTCDateTime(), nullable=True),
            sa.Column('acknowledged_at', UTCDateTime(), nullable=True),
            sa.Column('completed_at', UTCDateTime(), nullable=True),
            sa.Column('expires_at', UTCDateTime(), nullable=True),
            sa.Column('result', sa.JSON(), nullable=True),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('id', sa.Integer(), nullable=False),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index(
            op.f('ix_worker_commands_command_id'),
            'worker_commands',
            ['command_id'],
            unique=True,
        )
        op.create_index(
            op.f('ix_worker_commands_worker_id'),
            'worker_commands',
            ['worker_id'],
            unique=False,
        )
        op.create_index(
            op.f('ix_worker_commands_state'),
            'worker_commands',
            ['state'],
            unique=False,
        )


def downgrade() -> None:
    if table_exists('worker_commands'):
        op.drop_index(op.f('ix_worker_commands_state'), table_name='worker_commands')
        op.drop_index(
            op.f('ix_worker_commands_worker_id'), table_name='worker_commands'
        )
        op.drop_index(
            op.f('ix_worker_commands_command_id'), table_name='worker_commands'
        )
        op.drop_table('worker_commands')

    if table_exists('worker_sessions'):
        op.drop_index(
            op.f('ix_worker_sessions_worker_id'), table_name='worker_sessions'
        )
        op.drop_index(
            op.f('ix_worker_sessions_session_id'), table_name='worker_sessions'
        )
        op.drop_table('worker_sessions')

    with op.batch_alter_table('workers', schema=None) as batch_op:
        if column_exists('workers', 'reachability_mode'):
            batch_op.drop_column('reachability_mode')
        if column_exists('workers', 'capabilities'):
            batch_op.drop_column('capabilities')
