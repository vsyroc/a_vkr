import os
import pathlib

import torch

BASE_DIR = pathlib.Path(__file__).parent

# Пути к директориям
UPLOADS_DIR = pathlib.Path(os.getenv("UPLOADS_DIR", BASE_DIR / "uploads"))
PROCESSED_DIR = pathlib.Path(os.getenv("PROCESSED_DIR", BASE_DIR / "processed"))

# Модель
CHECKPOINT = BASE_DIR / "checkpoints" / "train_phosphorusV11_2_1200.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2

# Инференс
CROP_SIZE = 3000
