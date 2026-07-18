FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:${PATH} \
    PORT=8000 \
    FOOTEE_STORAGE_DIR=/tmp/footee-vision \
    LOW_MEMORY_MODE=true \
    SCENE_DETECTION_METHOD=hybrid \
    CORS_ORIGINS=https://footee-highlights.vercel.app \
    YOLO_MODEL_PATH=yolo11n.pt \
    TRACKING_MODEL_PATH=yolo11n.pt \
    TRACKING_BATCH_SIZE=4 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    MALLOC_ARENA_MAX=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. Uploads, thumbnails, and generated metadata are
# written only to the container's temporary filesystem.
RUN useradd --create-home --uid 1000 user \
    && mkdir -p /home/user/app/backend \
    && chown -R user:user /home/user/app

WORKDIR /home/user/app/backend

COPY backend/requirements.txt ./requirements.txt

# Install CPU-only PyTorch explicitly. This avoids pulling CUDA runtime wheels
# into a CPU Space and leaves more disk and memory for video processing.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && python -m pip install -r requirements.txt

COPY --chown=user:user backend/ ./

USER user

# Download the small detector into BACKEND_DIR while the image is being built.
# The first visitor therefore waits only for model initialization, not a model
# download, and detection and tracking share the same cached model instance.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" \
    && mkdir -p /tmp/footee-vision

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
