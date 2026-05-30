import torch
from saicinpainting.training.trainers import load_checkpoint

model = load_checkpoint("path/to/lama/checkpoint.ckpt", strict=False)
model.eval()
scripted = torch.jit.script(model)
scripted.save("checkpoints/lama.pt")
