FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WAN_MOVE_CKPT_DIR=/models/Wan-Move-14B-480P \
    WAN_MOVE_OUTPUT_DIR=/outputs \
    WAN_MOVE_HOST=0.0.0.0 \
    WAN_MOVE_PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgl1 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja \
    && grep -v '^flash_attn' requirements.txt > /tmp/requirements-no-flash.txt \
    && pip install -r /tmp/requirements-no-flash.txt \
    && pip install flash-attn --no-build-isolation \
    && pip install fastapi "uvicorn[standard]" python-multipart \
    && pip install xfuser --no-deps \
    && python -c "import decord, diffusers; assert diffusers.__version__ == '0.31.0', diffusers.__version__; print('decord', decord.__version__, 'diffusers', diffusers.__version__)"

COPY . .

RUN mkdir -p /models/Wan-Move-14B-480P /outputs

EXPOSE 8000

CMD ["torchrun", "--nproc_per_node=2", "api_server.py"]
