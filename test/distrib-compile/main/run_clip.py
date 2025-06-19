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
img = torch.randn(
    batch_size, 3, clip_config["image_resolution"], clip_config["image_resolution"]
)
txt = torch.randint(
    0, clip_config["vocab_size"], (batch_size, clip_config["context_length"])
)
labels = torch.arange(batch_size)


def loss_fn(logits_per_image, labels):
    loss_img = torch.nn.functional.cross_entropy(logits_per_image.T, labels)
    loss_txt = torch.nn.functional.cross_entropy(logits_per_image, labels)
    return (loss_img + loss_txt) / 2.0


# time microbatch schedules

from .clip_schedules import build_gpipe_schedule, print_schedule
from torch._dynamo.scheduling import DAGEdge, execute_schedule
import time

warmup = 3
iters = 10
num_mbs = 4

out = compiled(img, txt, dynamo_mb=42)
stg1 = compiled._ray_actors[0]
stg2 = compiled._ray_actors[1]
stg3 = compiled._ray_actors[2]


# "1F1B" schedule


def iter_1f1b_manual():
    done_stg1 = []
    done_stg2 = []
    done_stg3 = []

    for mb in range(num_mbs):
        out_ref = compiled(img, txt, dynamo_mb=mb)
        print(f"Calling backward stage 2 mb {mb}")
        grad3_img, grad3_txt = stg3.backward.options(num_returns=2).remote(
            mb, out_ref.get_ref(), truth=labels, loss_fn=loss_fn
        )
        print(f"Calling backward stage 1 mb {mb}")
        grad2 = stg2.backward.remote(mb, grad3_txt)
        print(f"Calling backward stage 0 mb {mb}")
        grad1 = stg1.backward.remote(mb, grad3_img)
        done_stg1.append(grad1)
        done_stg2.append(grad2)
        done_stg3.append(grad3_img)
        done_stg3.append(grad3_txt)

    upd3 = stg3.update.remote(*done_stg3)
    upd2 = stg2.update.remote(*done_stg2)
    upd1 = stg1.update.remote(*done_stg1)
    ray.get([upd3, upd2, upd1])


for _ in range(warmup):
    iter_1f1b_manual()

start = time.perf_counter()
for _ in range(iters):
    iter_1f1b_manual()
end = time.perf_counter()

print(
    f"1F1B manual schedule throughput: {(iters * num_mbs * batch_size)/(end - start):.2f} samples/sec"
)


# "GPIPE" schedule


# manual schedule
def iter_gpipe_manual():
    fwd_refs = []
    done_stg1 = []
    done_stg2 = []
    done_stg3 = []

    for mb in range(num_mbs):
        out_ref = compiled(img, txt, dynamo_mb=mb)
        fwd_refs.append(out_ref)

    for mb, out_ref in enumerate(fwd_refs):
        print(f"Calling backward stage 2 mb {mb}")
        grad3_img, grad3_txt = stg3.backward.options(num_returns=2).remote(
            mb, out_ref.get_ref(), truth=labels, loss_fn=loss_fn
        )
        print(f"Calling backward stage 1 mb {mb}")
        grad2 = stg2.backward.remote(mb, grad3_txt)
        print(f"Calling backward stage 0 mb {mb}")
        grad1 = stg1.backward.remote(mb, grad3_img)
        done_stg1.append(grad1)
        done_stg2.append(grad2)
        done_stg3.append(grad3_img)
        done_stg3.append(grad3_txt)

    upd3 = stg3.update.remote(*done_stg3)
    upd2 = stg2.update.remote(*done_stg2)
    upd1 = stg1.update.remote(*done_stg1)
    ray.get([upd3, upd2, upd1])


# build high-level schedule
schedule = build_gpipe_schedule(num_mbs, 3)
dag_edges = [DAGEdge(0, 2), DAGEdge(1, 2)]


def iter_gpipe():
    out = execute_schedule(compiled, schedule, dag_edges, [img, txt], labels, loss_fn)
    ray.get(out)


# warmup
for _ in range(warmup):
    iter_gpipe()
    iter_gpipe_manual()

# time
start = time.perf_counter()
for _ in range(iters):
    iter_gpipe()
end = time.perf_counter()

print(
    f"GPipe execute_schedule throughput: {(iters * batch_size)/(end - start):.0f} samples/sec"
)

# time
start = time.perf_counter()
for _ in range(iters):
    iter_gpipe_manual()
end = time.perf_counter()

print(
    f"GPipe manual schedule throughput: {(iters * batch_size)/(end - start):.0f} samples/sec"
)


# baseline: no distribution

from models.clip_baseline import CLIP as BaseCLIP

baseline = BaseCLIP(**clip_config)
baseline = torch.compile(baseline, distribute=False)
optim = torch.optim.Adam(model.parameters())


def iter_baseline():
    out = baseline(img, txt)
    loss = loss_fn(out, labels)
    loss.backward()
    optim.step()
    optim.zero_grad()


for _ in range(warmup):
    iter_baseline()

start_base = time.perf_counter()
for _ in range(iters):
    iter_baseline()
end_base = time.perf_counter()

print(
    f"Baseline throughput: {(iters * batch_size)/(end_base - start_base):.2f} samples/sec"
)
