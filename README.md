# Thai ALPR System - Automatic License Plate Recognition

A production-ready Thai Automatic License Plate Recognition system optimized for NVIDIA RTX 3060 GPU.

## 🌟 Features

### Core Capabilities
- **Real-time RTSP stream processing** with polygon zone detection
- **Dual-model AI pipeline**: Vehicle detection → License plate detection
- **ByteTrack/BoT-SORT** object tracking with deduplication
- **PaddleOCR** Thai language recognition with post-processing
- **TensorRT optimization** for maximum RTX 3060 performance
- **Active learning** support for continuous model improvement

### Processing Pipeline
```
RTSP Stream → Vehicle Detection (vehicles.pt) → Object Tracking → 
Zone Check → Plate Detection (best.pt) → OCR (PaddleOCR) → 
Post-Processing → Database Storage → WebSocket Broadcast
```

### Confidence-Based Logic
- **Confidence > 0.95**: Auto-approve (ALPR_AUTO) - editable
- **Confidence ≤ 0.95**: Manual verification queue (PENDING_VERIFY)
- **Manual corrections**: Saved as MLPR with active learning support

## 🏗️ Architecture

### Technology Stack
**Backend**
- FastAPI (Python 3.10+)
- PostgreSQL 15 (persistent data)
- Redis 7 (cache/queue/deduplication)
- YOLOv8 + TensorRT (detection)
- PaddleOCR (Thai OCR)
- ByteTrack (object tracking)

**Frontend**
- React 18 + Vite 5
- Tailwind CSS 3
- WebSocket (real-time updates)
- React Query (state management)

**Infrastructure**
- Docker + Docker Compose
- NVIDIA Container Runtime
- Nginx (reverse proxy)

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 3060 (12GB VRAM recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB+ for models and data

### Software Requirements
- **OS**: Ubuntu 20.04+ or Windows 11 with WSL2
- **CUDA**: 11.8 or higher
- **cuDNN**: 8.6 or higher
- **TensorRT**: 8.6 or higher
- **Docker**: 24.0+ with NVIDIA Container Runtime
- **Docker Compose**: 2.20+
- **Python**: 3.10+
- **Node.js**: 18+ (for frontend)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/hexter2018/thai-alpr-system.git
cd thai-alpr-system
```

### 2. Install NVIDIA Drivers & CUDA
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA Toolkit 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Install cuDNN
# Download from: https://developer.nvidia.com/cudnn
# Follow installation guide

# Verify CUDA
nvcc --version
```

### 3. Install TensorRT (Critical for RTX 3060)
```bash
# Download TensorRT 8.6 from NVIDIA Developer
# https://developer.nvidia.com/tensorrt

# Install (Debian/Ubuntu)
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-11.8_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-*/nv-tensorrt-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt

# Verify installation
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

### 4. Install Docker with NVIDIA Support
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 5. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env

# Key settings to update:
# - POSTGRES_PASSWORD
# - REDIS_PASSWORD
# - SECRET_KEY
# - RTSP_URL_1 (your camera URL)
# - CAMERA_ID_1
```

### 6. Prepare AI Models

#### Option A: Use Pre-trained Models
```bash
# Create models directory
mkdir -p backend/models

# Download models
# vehicles.pt - Vehicle detection model
# best.pt - License plate detection model
# Place models in backend/models/
```

#### Option B: Convert to TensorRT (Recommended)
```bash
# Install backend dependencies first
cd backend
pip install -r requirements.txt

# Export to TensorRT
python -c "
from ultralytics import YOLO

# Export vehicle model
model = YOLO('models/vehicles.pt')
model.export(format='engine', imgsz=640, half=True, workspace=4, device=0)

# Export plate model
model = YOLO('models/best.pt')
model.export(format='engine', imgsz=640, half=True, workspace=4, device=0)
"

# This creates:
# - models/vehicles.engine
# - models/best.engine
```

### 7. Initialize Database
```bash
# Start database services only
docker-compose up -d postgres redis

# Wait for services to be ready
sleep 10

# Run migrations
cd backend
pip install alembic
alembic upgrade head

# Optional: Seed sample data
python scripts/seed_data.py
```

### 8. Launch Full System
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend

# Services will be available at:
# - Backend API: http://localhost:8000
# - Frontend: http://localhost:3000
# - PgAdmin: http://localhost:5050 (development)
```

### 9. Access Dashboard
```
Open browser: http://localhost:3000

Default credentials (if authentication enabled):
Username: admin
Password: admin123
```

## 📁 Project Structure

```
thai-alpr-system/
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── core/           # AI processing pipeline
│   │   │   ├── ai_core.py  # Main orchestrator
│   │   │   ├── vehicle_detector.py
│   │   │   ├── plate_detector.py
│   │   │   ├── ocr_engine.py
│   │   │   ├── tracker.py
│   │   │   └── zone_manager.py
│   │   ├── api/            # FastAPI routes
│   │   ├── services/       # Business logic
│   │   ├── models.py       # Database models
│   │   └── schemas.py      # Pydantic schemas
│   ├── models/             # AI model files
│   │   ├── vehicles.pt
│   │   ├── vehicles.engine
│   │   ├── best.pt
│   │   └── best.engine
│   └── storage/            # File storage
├── frontend/               # React application
│   └── src/
│       ├── components/
│       │   ├── Dashboard/
│       │   ├── LiveMonitor/
│       │   ├── Verification/
│       │   └── MasterData/
│       └── services/
└── docker-compose.yml
```

## 🔧 Configuration

### Camera Configuration
```python
# Add camera via API or database
POST /api/cameras
{
    "camera_id": "entrance_camera_1",
    "camera_name": "Main Entrance",
    "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream1",
    "polygon_zone": [
        {"x": 100, "y": 200},
        {"x": 500, "y": 200},
        {"x": 500, "y": 600},
        {"x": 100, "y": 600}
    ],
    "frame_skip": 2,
    "min_confidence_vehicle": 0.5,
    "min_confidence_plate": 0.4,
    "dedup_window_seconds": 60
}
```

### Processing Parameters
```bash
# .env configuration
FRAME_SKIP=2                    # Process every N frames
MIN_VEHICLE_CONFIDENCE=0.5      # Vehicle detection threshold
MIN_PLATE_CONFIDENCE=0.4        # Plate detection threshold
HIGH_CONFIDENCE_THRESHOLD=0.95  # Auto-approve threshold
DEDUP_WINDOW_SECONDS=60         # Redis TTL for tracking IDs
```

## 🎯 API Endpoints

### ALPR Processing
```bash
# Upload image for processing
POST /api/alpr/process-image
Content-Type: multipart/form-data
{
    "file": <image_file>
}

# Get detection by ID
GET /api/alpr/detections/{log_id}

# Get pending verifications
GET /api/alpr/pending?page=1&limit=20

# Manual verification
POST /api/alpr/verify/{log_id}
{
    "corrected_plate": "กก1234",
    "corrected_province": "กรุงเทพมหานคร",
    "status": "MLPR",
    "verified_by": "operator_1"
}
```

### Master Vehicles
```bash
# List vehicles
GET /api/vehicles?page=1&limit=20

# Create vehicle
POST /api/vehicles
{
    "license_plate": "กก1234",
    "province": "กรุงเทพมหานคร",
    "vehicle_type": "car",
    "owner_name": "John Doe",
    "is_authorized": true
}

# Update vehicle
PUT /api/vehicles/{vehicle_id}

# Delete vehicle
DELETE /api/vehicles/{vehicle_id}
```

### Live Stream
```bash
# WebSocket connection
WS /api/stream/ws/{camera_id}

# Start processing
POST /api/stream/start/{camera_id}

# Stop processing
POST /api/stream/stop/{camera_id}
```

### Statistics
```bash
# Dashboard KPIs
GET /api/stats/dashboard

# Date range stats
GET /api/stats/range?start_date=2024-01-01&end_date=2024-01-31
```

## 🧪 Testing

### Unit Tests
```bash
cd backend
pytest tests/ -v --cov=app --cov-report=html
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### Load Testing
```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## 🔍 Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Restart Docker
sudo systemctl restart docker
```

### TensorRT Issues
```bash
# Check TensorRT installation
python -c "import tensorrt as trt; print(trt.__version__)"

# Rebuild engine files
python scripts/rebuild_tensorrt.py

# Check model compatibility
python scripts/verify_models.py
```

### OCR Low Accuracy
```bash
# Update PaddleOCR
pip install --upgrade paddleocr

# Use preprocessing
# Edit backend/app/core/ocr_engine.py
# Adjust preprocessing parameters

# Check plate image quality
# Ensure minimum 40px height
# Check blur detection threshold
```

### RTSP Connection Failed
```bash
# Test RTSP URL manually
ffplay rtsp://admin:password@192.168.1.100:554/stream1

# Check network connectivity
ping 192.168.1.100

# Verify credentials
# Check camera documentation

# Adjust timeout settings
# Edit backend/app/services/rtsp_service.py
```

## 📊 Performance Optimization

### TensorRT Optimization
- **FP16 mode**: 2-3x faster than FP32, minimal accuracy loss
- **INT8 mode**: 3-5x faster, requires calibration
- **Batch processing**: Group detections for better throughput

### Redis Optimization
```bash
# Increase maxmemory
redis-cli CONFIG SET maxmemory 4gb

# Adjust eviction policy
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

### Database Optimization
```sql
-- Add indexes for common queries
CREATE INDEX idx_detection_timestamp ON access_logs(detection_timestamp DESC);
CREATE INDEX idx_tracking_camera ON access_logs(tracking_id, camera_id);

-- Partition large tables
CREATE TABLE access_logs_2024_01 PARTITION OF access_logs
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

## 🛡️ Security Best Practices

1. **Change default passwords** in `.env`
2. **Enable HTTPS** with SSL certificates
3. **Implement authentication** for API endpoints
4. **Use API keys** for external integrations
5. **Regular backups** of database
6. **Rate limiting** on API endpoints
7. **Input validation** for all user inputs
8. **Secure RTSP credentials** (use environment variables)

## 📈 Monitoring

### System Metrics
```bash
# GPU monitoring
watch -n 1 nvidia-smi

# Docker stats
docker stats

# Application logs
docker-compose logs -f --tail=100 backend
```

### Application Health
```bash
# Health check endpoint
curl http://localhost:8000/health

# Metrics endpoint (Prometheus)
curl http://localhost:8000/metrics
```

## 🔄 Active Learning & Retraining

### Collect Training Data
```bash
# Manual corrections automatically saved to:
backend/storage/dataset/train/

# Check dataset size
python scripts/check_dataset.py

# When ready (100+ samples recommended):
python scripts/retrain_model.py --model plate --epochs 50
```

## 📝 License

Copyright © 2024. All rights reserved.

## 🤝 Support

For issues and questions:
- GitHub Issues: [link]
- Email: support@example.com
- Documentation: [link]

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 implementation
- **PaddleOCR**: Thai OCR engine
- **ByteTrack**: Object tracking algorithm
- **NVIDIA**: TensorRT optimization framework