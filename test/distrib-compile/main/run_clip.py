import ray
import torch
from torch import nn, optim
from models.clip import CLIP

# torch._dynamo.reset()

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
compiled = torch.compile(model, distribute=True)

batch_size = 100
img = torch.randn(batch_size, 3, clip_config["image_resolution"], clip_config["image_resolution"])
txt = torch.randint(0, clip_config["vocab_size"], (batch_size, clip_config["context_length"]))

# img_out, txt_out = model(img, txt)
# print(torch.sum(img_out))
# img_out, txt_out = compiled(img, txt)
# print(torch.sum(img_out))

# test microbatching

def loss_fn(logits_per_image):
    labels = torch.arange(logits_per_image.size(0))
    loss_img = torch.nn.functional.cross_entropy(logits_per_image.T, labels)
    loss_txt = torch.nn.functional.cross_entropy(logits_per_image, labels)
    return (loss_img + loss_txt)/2.0 

out = compiled(img, txt, dynamo_mb=42)
stg1 = compiled._ray_actors[1]
stg2 = compiled._ray_actors[2]
stg3 = compiled._ray_actors[3]

out = compiled(img, txt, dynamo_mb=42)
# grad3_img, grad3_txt = stg3.backward.remote(0, out.get_ref(), truth=y, loss_fn=loss_fn)
# grad2 = stg2.backward.remote(0, grad3_txt)
# grad1 = stg1.backward.remote(0, grad3_img)

# upd3 = stg3.update.remote(grad3_img, grad3_txt)
# upd2 = stg2.update.remote(grad2)
# upd1 = stg1.update.remote(grad1)
# ray.get([upd3, upd2, upd1])