FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# PyTorch отдельно — самый тяжёлый слой, кешируется Docker-ом
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Остальные зависимости (без torch — уже стоит)
COPY pyproject.toml ./
RUN pip3 install --no-cache-dir . || pip3 install --no-cache-dir \
    fastapi uvicorn[standard] python-multipart jinja2 loguru \
    numpy pillow rasterio opencv-python-headless albumentations
RUN pip3 install --no-cache-dir "numpy<2"    

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]