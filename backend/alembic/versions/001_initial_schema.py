"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2024-02-16

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('master_vehicles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('license_plate', sa.String(length=20), nullable=False),
        sa.Column('province', sa.String(length=100), nullable=False),
        sa.Column('vehicle_type', sa.Enum('car', 'truck', 'motorcycle', 'bus', 'van', 'unknown', name='vehicletype'), nullable=True),
        sa.Column('brand', sa.String(length=100), nullable=True),
        sa.Column('model', sa.String(length=100), nullable=True),
        sa.Column('color', sa.String(length=50), nullable=True),
        sa.Column('year', sa.Integer(), nullable=True),
        sa.Column('owner_name', sa.String(length=255), nullable=True),
        sa.Column('owner_phone', sa.String(length=20), nullable=True),
        sa.Column('owner_address', sa.Text(), nullable=True),
        sa.Column('is_authorized', sa.Boolean(), nullable=True),
        sa.Column('is_blacklisted', sa.Boolean(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('created_by', sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_plate_province', 'master_vehicles', ['license_plate', 'province'])
    op.create_index('idx_authorized', 'master_vehicles', ['is_authorized'])
    op.create_index(op.f('ix_master_vehicles_license_plate'), 'master_vehicles', ['license_plate'], unique=True)
    
    op.create_table('camera_configs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('camera_id', sa.String(length=100), nullable=False),
        sa.Column('camera_name', sa.String(length=255), nullable=False),
        sa.Column('rtsp_url', sa.String(length=500), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('polygon_zone', postgresql.JSONB(), nullable=True),
        sa.Column('frame_skip', sa.Integer(), nullable=True),
        sa.Column('min_confidence_vehicle', sa.Float(), nullable=True),
        sa.Column('min_confidence_plate', sa.Float(), nullable=True),
        sa.Column('dedup_window_seconds', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_camera_configs_camera_id'), 'camera_configs', ['camera_id'], unique=True)
    
    op.create_table('system_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('metric_date', sa.DateTime(), nullable=True),
        sa.Column('total_detections', sa.Integer(), nullable=True),
        sa.Column('alpr_auto_count', sa.Integer(), nullable=True),
        sa.Column('mlpr_count', sa.Integer(), nullable=True),
        sa.Column('pending_count', sa.Integer(), nullable=True),
        sa.Column('rejected_count', sa.Integer(), nullable=True),
        sa.Column('average_confidence', sa.Float(), nullable=True),
        sa.Column('high_confidence_percentage', sa.Float(), nullable=True),
        sa.Column('avg_processing_time_ms', sa.Float(), nullable=True),
        sa.Column('unique_vehicles_detected', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_metric_date', 'system_metrics', ['metric_date'])
    
    op.create_table('access_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tracking_id', sa.String(length=50), nullable=True),
        sa.Column('detection_timestamp', sa.DateTime(), nullable=False),
        sa.Column('camera_id', sa.String(length=100), nullable=True),
        sa.Column('full_image_path', sa.String(length=500), nullable=True),
        sa.Column('plate_crop_path', sa.String(length=500), nullable=True),
        sa.Column('detected_plate', sa.String(length=20), nullable=True),
        sa.Column('detected_province', sa.String(length=100), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('vehicle_type', sa.Enum('car', 'truck', 'motorcycle', 'bus', 'van', 'unknown', name='vehicletype'), nullable=True),
        sa.Column('vehicle_bbox', postgresql.JSONB(), nullable=True),
        sa.Column('plate_bbox', postgresql.JSONB(), nullable=True),
        sa.Column('status', sa.Enum('ALPR_AUTO', 'PENDING_VERIFY', 'MLPR', 'REJECTED', name='processstatus'), nullable=True),
        sa.Column('corrected_plate', sa.String(length=20), nullable=True),
        sa.Column('corrected_province', sa.String(length=100), nullable=True),
        sa.Column('verified_at', sa.DateTime(), nullable=True),
        sa.Column('verified_by', sa.String(length=100), nullable=True),
        sa.Column('verification_notes', sa.Text(), nullable=True),
        sa.Column('added_to_training', sa.Boolean(), nullable=True),
        sa.Column('vehicle_id', sa.Integer(), nullable=True),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('ocr_raw_output', postgresql.JSONB(), nullable=True),
        sa.Column('model_versions', postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(['vehicle_id'], ['master_vehicles.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_camera_timestamp', 'access_logs', ['camera_id', 'detection_timestamp'])
    op.create_index('idx_status_timestamp', 'access_logs', ['status', 'detection_timestamp'])
    op.create_index('idx_tracking_timestamp', 'access_logs', ['tracking_id', 'detection_timestamp'])
    op.create_index(op.f('ix_access_logs_detected_plate'), 'access_logs', ['detected_plate'])
    op.create_index(op.f('ix_access_logs_detection_timestamp'), 'access_logs', ['detection_timestamp'])
    op.create_index(op.f('ix_access_logs_status'), 'access_logs', ['status'])
    op.create_index(op.f('ix_access_logs_tracking_id'), 'access_logs', ['tracking_id'])
    op.create_index(op.f('ix_access_logs_vehicle_id'), 'access_logs', ['vehicle_id'])

def downgrade():
    op.drop_table('access_logs')
    op.drop_table('system_metrics')
    op.drop_table('camera_configs')
    op.drop_table('master_vehicles')
    op.execute('DROP TYPE IF EXISTS vehicletype')
    op.execute('DROP TYPE IF EXISTS processstatus')
