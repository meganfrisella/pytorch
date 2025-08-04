import ray
import torch
from torch import nn, optim
from models.clip import CLIP
import time

ray.init(include_dashboard=True, namespace="clip")
torch.manual_seed(0)

# clip_config = {
#     "embed_dim": 2048,
#     "image_resolution": 224,
#     "vision_layers": 16,
#     "vision_width": 256,
#     "vision_patch_size": 32,
#     "context_length": 32,
#     "vocab_size": 49408,
#     "transformer_width": 1024,
#     "transformer_heads": 8,
#     "transformer_layers": 16,
# }
clip_config = {
    "embed_dim": 128,
    "image_resolution": 224,
    "vision_layers": 2,
    "vision_width": 128,
    "vision_patch_size": 32,
    "context_length": 10,
    "vocab_size": 49408,
    "transformer_width": 128,
    "transformer_heads": 8,
    "transformer_layers": 2,
}

batch_size = 128
warmup = 10
iters = 100
num_mbs = 4

img = torch.randn(
    batch_size, 3, clip_config["image_resolution"], clip_config["image_resolution"]
).to('cuda')
txt = torch.randint(
    0, clip_config["vocab_size"], (batch_size, clip_config["context_length"])
).to('cuda')
labels = torch.arange(batch_size).to('cuda')

def loss_fn(logits_per_image, labels):
    loss_img = torch.nn.functional.cross_entropy(logits_per_image.T, labels)
    loss_txt = torch.nn.functional.cross_entropy(logits_per_image, labels)
    return (loss_img + loss_txt) / 2.0

# time microbatch schedules

model = CLIP(**clip_config)
model.to('cuda')

from .clip_schedules import build_gpipe_schedule, print_schedule
from torch._dynamo.scheduling import DAGEdge, execute_schedule
import time

from torch._dynamo.backends.debugging import eager
compiled = torch.compile(model, distribute=True, backend=eager)
compiled(img, txt, dynamo_mb=42).get()
torch._dynamo.eval_frame.dynamo_tls.currently_compiling = None
stg1 = compiled._ray_actors[0]
stg2 = compiled._ray_actors[1]
stg3 = compiled._ray_actors[2]

ray.get(stg1.send_input.remote(img))
ray.get(stg2.send_input.remote(txt))
ray.get(stg3.send_truth.remote(labels))

# "1F1B" schedule

def iter_1f1b_manual():
    done_stg1 = []
    done_stg2 = []
    done_stg3 = []

    out_ref = compiled(None, None, dynamo_mb=0)
    for mb in range(1, num_mbs):
        # print(f"Calling backward stage 2 mb {mb}")
        grad3_img, grad3_txt, done3_img, done3_txt = stg3.backward.options(num_returns=4).remote(
            mb-1, out_ref.get_ref(), loss_fn=loss_fn
        )
        out_ref = compiled(None, None, dynamo_mb=mb)
        # print(f"Calling backward stage 1 mb {mb}")
        grad2, done2 = stg2.backward.options(num_returns=2).remote(mb-1, grad3_txt)
        # print(f"Calling backward stage 0 mb {mb}")
        grad1, done1 = stg1.backward.options(num_returns=2).remote(mb-1, grad3_img)
        done_stg1.append(done1)
        done_stg2.append(done2)
        done_stg3.append(done3_img)
        done_stg3.append(done3_txt)
    
    grad3_img, grad3_txt, done3_img, done3_txt = stg3.backward.options(num_returns=4).remote(
        num_mbs-1, out_ref.get_ref(), loss_fn=loss_fn
    )
    grad2, done2 = stg2.backward.options(num_returns=2).remote(num_mbs-1, grad3_txt)
    grad1, done1 = stg1.backward.options(num_returns=2).remote(num_mbs-1, grad3_img)
    done_stg1.append(done1)
    done_stg2.append(done2)
    done_stg3.append(done3_img)
    done_stg3.append(done3_txt)

    upd3 = stg3.update.remote(*done_stg3)
    upd2 = stg2.update.remote(*done_stg2)
    upd1 = stg1.update.remote(*done_stg1)
    # ray.wait([upd3, upd2, upd1], fetch_local=False)
    ray.get([upd3, upd2, upd1])

# build high-level schedule
from torch._dynamo.scheduling import Task
schedule = [
    [Task(stage_id=0, mb_idx=0, is_fwd=True), None,                                    Task(stage_id=0, mb_idx=1, is_fwd=True),  Task(stage_id=0, mb_idx=0, is_fwd=False), Task(stage_id=0, mb_idx=2, is_fwd=True),  Task(stage_id=0, mb_idx=1, is_fwd=False), Task(stage_id=0, mb_idx=3, is_fwd=True),  Task(stage_id=0, mb_idx=2, is_fwd=False), None,                                     Task(stage_id=0, mb_idx=3, is_fwd=False), None], 
    [Task(stage_id=1, mb_idx=0, is_fwd=True), None,                                    Task(stage_id=1, mb_idx=1, is_fwd=True),  Task(stage_id=1, mb_idx=0, is_fwd=False), Task(stage_id=1, mb_idx=2, is_fwd=True),  Task(stage_id=1, mb_idx=1, is_fwd=False), Task(stage_id=1, mb_idx=3, is_fwd=True),  Task(stage_id=1, mb_idx=2, is_fwd=False), None,                                     Task(stage_id=1, mb_idx=3, is_fwd=False), None], 
    [None,                                    Task(stage_id=2, mb_idx=0, is_fwd=True), Task(stage_id=2, mb_idx=0, is_fwd=False), Task(stage_id=2, mb_idx=1, is_fwd=True),  Task(stage_id=2, mb_idx=1, is_fwd=False), Task(stage_id=2, mb_idx=2, is_fwd=True),  Task(stage_id=2, mb_idx=2, is_fwd=False), Task(stage_id=2, mb_idx=3, is_fwd=True),  Task(stage_id=2, mb_idx=3, is_fwd=False), None, None]]
print_schedule(schedule)

def iter_1f1b():
    out = execute_schedule(compiled, schedule, [None, None], None, loss_fn)
    ray.get(out)

for _ in range(warmup):
    iter_1f1b()

start = time.perf_counter()
for _ in range(iters):
    iter_1f1b()
end = time.perf_counter()

print(
    f"1F1B execute_schedule throughput: {(iters * num_mbs * batch_size)/(end - start):.0f} samples/sec"
)
print(
    f"1F1B time: {(end - start)*1000000/iters:.0f} us"
)


for _ in range(warmup):
    iter_1f1b_manual()

start = time.perf_counter()
for _ in range(iters):
    iter_1f1b_manual()
end = time.perf_counter()

print(
    f"1F1B manual schedule throughput: {(iters * num_mbs * batch_size)/(end - start):.0f} samples/sec"
)
print(
    f"1F1B time: {(end - start)*1000000/iters:.0f} us"
)

# "GPIPE" schedule


# manual schedule
def iter_gpipe_manual():
    fwd_refs = []
    done_stg1 = []
    done_stg2 = []
    done_stg3 = []

    for mb in range(num_mbs):
        # print(f"Fwd mb {mb}")
        out_ref = compiled(None, None, dynamo_mb=mb)
        fwd_refs.append(out_ref)

    for mb, out_ref in enumerate(fwd_refs):
        # print(f"Bwd 2:{mb}")
        grad3_img, grad3_txt, done_img, done_txt = stg3.backward.options(num_returns=4).remote(
            mb, out_ref.get_ref(), loss_fn=loss_fn
        )
        # print(f"Bwd 1:{mb}")
        grad2, done2 = stg2.backward.options(num_returns=2).remote(mb, grad3_txt)
        # print(f"Bwd 0:{mb}")
        grad1, done1 = stg1.backward.options(num_returns=2).remote(mb, grad3_img)
        done_stg1.append(done1)
        done_stg2.append(done2)
        done_stg3.append(done_img)
        done_stg3.append(done_txt)

    upd3 = stg3.update.remote(*done_stg3)
    upd2 = stg2.update.remote(*done_stg2)
    upd1 = stg1.update.remote(*done_stg1)
    ray.get([upd3, upd2, upd1])


# build high-level schedule
schedule = build_gpipe_schedule(num_mbs, 3)
print_schedule(schedule)

def iter_gpipe():
    out = execute_schedule(compiled, schedule, [None, None], None, loss_fn)
    ray.get(out)

# warmup
for _ in range(warmup):
    iter_gpipe()

# time
start = time.perf_counter()
for _ in range(iters):
    iter_gpipe()
end = time.perf_counter()

print(
    f"GPipe execute_schedule throughput: {(iters * num_mbs * batch_size)/(end - start):.0f} samples/sec"
)
print(
    f"GPipe execute_schedule time: {(end - start)*1000000/iters:.0f} us"
)

# warmup
for _ in range(warmup):
    iter_gpipe_manual()

start = time.perf_counter()
for _ in range(iters):
    iter_gpipe_manual()
end = time.perf_counter()

print(
    f"GPipe manual schedule throughput: {(iters * num_mbs * batch_size)/(end - start):.0f} samples/sec"
)
print(
    f"GPipe time: {(end - start)*1000000/iters:.0f} us"
)


# baseline: no distribution

from models.clip_baseline import CLIP as BaseCLIP
from torch._dynamo.backends.debugging import eager

baseline = BaseCLIP(**clip_config)
baseline.to('cuda')
# baseline = torch.compile(baseline, distribute=False)
optim = torch.optim.Adam(baseline.parameters())


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
    f"Baseline throughput: {(iters * batch_size)/(end_base - start_base):.0f} samples/sec"
)
print(
    f"Baseline time: {(end_base - start_base)*1000000/iters:.0f} us"
)