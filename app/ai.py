import pathlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from app.config import CHECKPOINT, DEVICE, CROP_SIZE
from simple_lama_inpainting import SimpleLama
from loguru import logger
import rasterio
from time import perf_counter

from app.networks.modeling import deeplabv3plus_mobilenet

_model = None
_lama = None


def get_model():
    global _model
    if _model is None:
        logger.info("Loading segmentation model...")
        _model = deeplabv3plus_mobilenet(num_classes=2)
        _model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
        _model = _model.to(DEVICE)
        _model.eval()
        logger.info(f"Segmentation model loaded. {DEVICE}")
    return _model


def get_lama():
    global _lama
    if _lama is None:
        logger.info("Loading LaMa model...")
        _lama = SimpleLama(device=DEVICE)
        logger.info("LaMa model loaded.")
    return _lama


def run_segmentation(model, input_path, mask_path, crop=CROP_SIZE):
    tif_path = input_path.with_suffix(".tif")
    img_pil = Image.open(input_path).convert("RGB")
    img_pil.save(tif_path, format="TIFF")

    with rasterio.open(tif_path) as src:
        w = min(crop, src.width)
        h = min(crop, src.height)
        image_arr = src.read(window=rasterio.windows.Window(0, 0, w, h)).astype(
            np.float32
        )

    tif_path.unlink(missing_ok=True)

    image_norm = (image_arr / 255.0 * 2) - 1

    t0 = perf_counter()
    with torch.no_grad():
        tensor = torch.from_numpy(image_norm[np.newaxis]).to(DEVICE)
        head = model(tensor)
        head = F.sigmoid(head)
    logger.info(f"Segmentation time: {perf_counter() - t0:.3f}s")

    head_np = head.cpu().numpy()[0, 0]
    mask_uint8 = (head_np * 255).astype(np.uint8)
    Image.fromarray(mask_uint8).save(mask_path)


def run_inpainting(input_path, mask_path, inpainted_path):
    lama = get_lama()

    orig_pil = Image.open(input_path).convert("RGB")
    mask_pil = Image.open(mask_path).convert("L")

    t0 = perf_counter()
    inpainted_pil = lama(orig_pil, mask_pil)
    logger.info(f"Inpainting time: {perf_counter() - t0:.3f}s")

    inpainted_pil.save(inpainted_path)
