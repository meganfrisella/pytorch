import ray
import torch
from torch import nn, optim
from models.llama import Transformer, LLAMA_DEBUG, LLAMA_1B

llama_config = LLAMA_DEBUG

loss_fn = torch.nn.CrossEntropyLoss()

model = Transformer(llama_config)
compiled = torch.compile(model, distribute=True)

batch_size = 100
seq_len = 32
x = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len))
y = torch.zeros((batch_size, llama_config.vocab_size), dtype=torch.long)

# time microbatch schedules

import time
warmup = 3
iters = 10
num_mbs = 4

out = compiled(x)
stg1 = compiled._ray_actors[1]
stg2 = compiled._ray_actors[2]


# "1F1B" schedule

def iter_1f1b():
    done_stg1 = []
    done_stg2 = []

    for mb in range(num_mbs):
        out_ref = compiled(x, dynamo_mb=mb)
        grad2 = stg2.backward.remote(
            mb, out_ref.get_ref(), truth=y, loss_fn=loss_fn
        )
        grad1 = stg1.backward.remote(mb, grad2)
        done_stg1.append(grad1)
        done_stg2.append(grad2)

    upd2 = stg2.update.remote(*done_stg2)
    upd1 = stg1.update.remote(*done_stg1)
    ray.get([upd2, upd1])


for _ in range(warmup):
    iter_1f1b()

start_1f1b = time.perf_counter()
for _ in range(iters):
    iter_1f1b()
end_1f1b = time.perf_counter()

print(
    f"1F1B throughput: {(iters * batch_size * num_mbs * seq_len)/(end_1f1b - start_1f1b):.0f} tokens/sec"
)


# "GPIPE" schedule

def iter_gpipe():
    fwd_refs = []
    done_stg1 = []
    done_stg2 = []

    for mb in range(num_mbs):
        out_ref = compiled(x, dynamo_mb=mb)
        fwd_refs.append(out_ref)

    for mb, out_ref in enumerate(fwd_refs):
        grad2 = stg2.backward.remote(
            mb, out_ref.get_ref(), truth=y, loss_fn=loss_fn
        )
        grad1 = stg1.backward.remote(mb, grad2)
        done_stg1.append(grad1)
        done_stg2.append(grad2)

    upd2 = stg2.update.remote(*done_stg2)
    upd1 = stg1.update.remote(*done_stg1)
    ray.get([upd2, upd1])


for _ in range(warmup):
    iter_gpipe()

start_gpipe = time.perf_counter()
for _ in range(iters):
    iter_gpipe()
end_gpipe = time.perf_counter()

print(
    f"GPipe throughput: {(iters * batch_size * num_mbs * seq_len)/(end_gpipe - start_gpipe):.0f} tokens/sec"
)


# baseline: no distribution

optim = torch.optim.Adam(model.parameters())

def iter_baseline():
    out = model(x)
    loss = loss_fn(out, y)
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
    f"Baseline throughput: {(iters * batch_size * seq_len)/(end_base - start_base):.0f} tokens/sec"
)