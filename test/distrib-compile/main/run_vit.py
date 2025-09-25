import torch
import ray
from models.vit import ViT
from torch._dynamo.piper import Task, piper_exec, piper_setup

torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch.manual_seed(0)

batch_size = 128
num_classes = 1000
warmup = 10
iters = 10

model = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = num_classes,
    dim = 1024,
    depth = 2,
    max_tokens_per_depth = (32, 32),
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
).to('cuda')

img = torch.randn(batch_size, 3, 256, 256).to('cuda')
labels = torch.randn((batch_size, num_classes))

# TODO: need to include kwargs here
model = piper_setup(model, [img, False], dynamic=True)

stg1 = model._ray_actors[0]
stg2 = model._ray_actors[1]

ray.get(stg1.send_input.remote(img))
ray.get(stg2.send_truth.remote(labels))


from .llama_schedules import build_1f1b_schedule

# meausre throughput

num_mbs = 4

# build high-level schedule
schedule = build_1f1b_schedule(num_mbs, 2)
loss_fn = torch.nn.CrossEntropyLoss()

def iter_1f1b():
    # TODO: need to include kwargs here
    out = piper_exec(model, schedule, [img, False], labels, loss_fn)
    ray.get(out)
    # ray.wait(out, fetch_local=False)

iter_1f1b()
exit()

# warmup
for _ in range(warmup):
    iter_1f1b()

ray.get(stg1.start_mem_tracking.remote())
ray.get(stg2.start_mem_tracking.remote())

iter_1f1b()

peak1 = ray.get(stg1.stop_mem_tracking.remote())
peak2 = ray.get(stg2.stop_mem_tracking.remote())

print(f"actor 0 peak mem: {peak1}")
print(f"actor 1 peak mem: {peak2}")

# time
start = time.perf_counter()
for _ in range(iters):
    iter_1f1b()
end = time.perf_counter()

print(
    f"1F1B throughput: {(iters * batch_size * num_mbs)/(end - start):.0f} samples/sec"
)
print(
    f"time: {(end - start)*1000000/iters:.0f} us"
)
