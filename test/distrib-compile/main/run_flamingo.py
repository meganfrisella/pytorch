from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor

vit = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
).to('cuda')

vit = Extractor(vit, return_embeddings_only = True).to('cuda')

# first take your trained image encoder and wrap it in an adapter that returns the image embeddings
# here we use the ViT from the vit-pytorch library

import torch
from models.flamingo import FlamingoPaLM
from torch._dynamo.backends.debugging import eager

# a PaLM language model, the 540 billion parameter model from google that shows signs of general intelligence

flamingo_palm = FlamingoPaLM(
    num_tokens = 20000,          # number of tokens
    dim = 1024,                  # dimensions
    depth = 12,                  # depth
    heads = 8,                   # attention heads
    dim_head = 64,               # dimension per attention head
    img_encoder = vit,           # plugin your image encoder (this can be optional if you pass in the image embeddings separately, but probably want to train end to end given the perceiver resampler)
    media_token_id = 3,          # the token id representing the [media] or [image]
    cross_attn_every = 3,        # how often to cross attend
    perceiver_num_latents = 64,  # perceiver number of latents, should be smaller than the sequence length of the image tokens
    perceiver_depth = 2          # perceiver resampler depth
).to('cuda')

text = torch.randint(0, 20000, (2, 512)).to('cuda')

out = flamingo_palm(text)
print(out[0][0])

# flamingo_palm = torch.compile(flamingo_palm, distribute=True, backend=eager)
flamingo_palm = torch.compile(flamingo_palm, distribute=True)
# flamingo_palm = torch.compile(flamingo_palm, backend=eager)

# train your PaLM as usual
palm_logits = flamingo_palm(text).get()
print(palm_logits.shape)
print(palm_logits[0][0])

exit()

# after much training off the regular PaLM logits
# now you are ready to train Flamingo + PaLM
# by passing in images, it automatically freezes everything but the perceiver and cross attention blocks, as in the paper

dialogue = torch.randint(0, 20000, (4, 512)).to('cuda')
images = torch.randn(4, 2, 3, 256, 256).to('cuda')

flamingo_logits = flamingo_palm(dialogue, images=images)
print(flamingo_logits.shape)

# do your usual cross entropy loss