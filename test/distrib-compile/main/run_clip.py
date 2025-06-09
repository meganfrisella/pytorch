import ray
import torch
from torch import nn, optim
from models.clip import CLIP

clip_config = {
    "embed_dim": 512,
    "image_resolution": 224,
    "vision_layers": 2,
    "vision_width": 256,
    "vision_patch_size": 32,
    "context_length": 77,
    "vocab_size": 49408,
    "transformer_width": 256,
    "transformer_heads": 8,
    "transformer_layers": 2,
}

model = CLIP(**clip_config)
compiled = torch.compile(model, distribute=False)

batch_size = 100
img = torch.randn(batch_size, 3, clip_config["image_resolution"], clip_config["image_resolution"])
txt = torch.randint(0, clip_config["vocab_size"], (batch_size, clip_config["context_length"]))

img_out, txt_out = model(img, txt)
img_out, txt_out = compiled(img, txt)
