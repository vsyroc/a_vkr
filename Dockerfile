FROM python:3.14-slim

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir \
    "fastapi>=0.136.3,<0.137.0" \
    "uvicorn[standard]>=0.48.0,<0.49.0" \
    "python-multipart>=0.0.29,<0.0.30" \
    "jinja2>=3.1.6,<4.0.0" \
    "loguru>=0.7.0" \
    "torch>=2.0.0,<3.0.0" \
    "numpy>=1.26.0,<3.0.0" \
    "pillow>=10.0.0,<12.0.0" \
    "rasterio>=1.3.0,<2.0.0" \
    "opencv-python-headless>=4.9.0,<5.0.0" \
    "albumentations>=1.4.0,<3.0.0"

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]