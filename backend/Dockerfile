FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Bangkok

WORKDIR /app

# ── System packages ───────────────────────────────────────────────────────────
# Install FFmpeg with H.265 (libx265) support + GStreamer RTSP plugins
# OpenCV will use FFmpeg backend to decode H.265 RTSP streams
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    # OpenCV display deps
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # FFmpeg with H.265 support (key package)
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavdevice-dev \
    # H.265 codec
    libx265-dev \
    libx264-dev \
    # RTSP / network
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tell OpenCV/FFmpeg to prefer TCP for RTSP (avoids UDP packet loss)
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/__init__.py    ./app/
COPY app/config.py      ./app/
COPY app/database.py    ./app/
COPY app/models.py      ./app/
COPY app/schemas.py     ./app/
COPY app/dependencies.py ./app/
COPY app/main.py        ./app/
COPY app/core/          ./app/core/
COPY app/services/      ./app/services/
COPY app/api/           ./app/api/
COPY app/utils/         ./app/utils/

RUN mkdir -p /app/models /app/storage/images /app/storage/plates /app/storage/dataset

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]