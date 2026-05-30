import pathlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from app.config import CHECKPOINT, DEVICE
from loguru import logger
import rasterio
from time import perf_counter

from app.networks.modeling import deeplabv3plus_mobilenet

# Модель загружается один раз при старте приложения
_model = None


def get_model():
    global _model
    if _model is None:
        logger.info("Loading model...")
        _model = deeplabv3plus_mobilenet(num_classes=2)
        _model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
        _model = _model.to(DEVICE)
        _model.eval()
        logger.info(f"Model loaded on {DEVICE}")
    return _model


def run_inference(
    model, input_path: pathlib.Path, output_path: pathlib.Path, crop: int = 3000
) -> None:
    """
    Принимает путь к изображению, запускает модель, сохраняет результат.
    """

    # Конвертируем входной файл в TIFF для gdal
    tif_path = input_path.with_suffix(".tif")
    img_pil = Image.open(input_path).convert("RGB")
    img_pil.save(tif_path, format="TIFF")

    # Читаем через rasterio
    with rasterio.open(tif_path) as src:
        w = min(crop, src.width)
        h = min(crop, src.height)
        image = src.read(window=rasterio.windows.Window(0, 0, w, h)).astype(np.float32)

    # Нормализация
    image = (image / 255.0 * 2) - 1

    t0 = perf_counter()
    with torch.no_grad():
        tensor = torch.from_numpy(image[np.newaxis]).to(DEVICE)
        head = model(tensor)
        head = F.sigmoid(head)
    logger.info(f"Inference time: {perf_counter() - t0:.3f}s")

    # Сохраняем результат
    head_np = head.cpu().numpy()[0, 0]
    head_rescaled = (head_np * 255).astype(np.uint8)
    Image.fromarray(head_rescaled).save(output_path)

    # Убираем временный .tif
    tif_path.unlink(missing_ok=True)
