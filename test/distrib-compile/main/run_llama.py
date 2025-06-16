import ray
import torch
from torch import nn, optim
from models.llama import Transformer, LLAMA_DEBUG, LLAMA_1B

llama_config = LLAMA_DEBUG

loss_fn = torch.nn.CrossEntropyLoss()

model = Transformer(llama_config)
compiled_model = torch.compile(model, distribute=True)

batch_size = 100
seq_len = 32
x = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len))
y = torch.zeros((batch_size, llama_config.vocab_size), dtype=torch.long)

# """
# test microbatching

out = compiled_model(x, mb=42)
stg1 = compiled_model._ray_actors[1]
stg2 = compiled_model._ray_actors[2]

done_stg1 = []
done_stg2 = []
for mb in range(2):
    out = compiled_model(x, mb=mb)
    grad2 = stg2.backward.remote(mb, out.get_ref(), truth=y, loss_fn=loss_fn)
    grad1 = stg1.backward.remote(mb, grad2)
    done_stg2.append(grad2)
    done_stg1.append(grad1)

upd2 = stg2.update.remote(*done_stg2)
upd1 = stg1.update.remote(*done_stg1)
ray.get([upd2, upd1])

# """

"""
# test backprop

mb_idx = 0
out = compiled_model(x, mb_idx=mb_idx)
stg1 = compiled_model._ray_actors[1]
stg2 = compiled_model._ray_actors[2]

grad2 = stg2.backward.remote(mb_idx, out.get_ref(), truth=y, loss_fn=loss_fn)
grad1 = stg1.backward.remote(mb_idx, grad2)
upd2 = stg2.update.remote(grad2)
upd1 = stg1.update.remote(grad1)
ray.get([upd2, upd1])

out = compiled_model(x, mb_idx=mb_idx)
print(torch.sum(out))

out = compiled_model(x, mb_idx=mb_idx)
print(torch.sum(out))

optim = optim.Adam(model.parameters())
out = model(x)
loss = loss_fn(out, y)
loss.backward()
optim.step()
optim.zero_grad()
out = model(x)
print(torch.sum(out))

"""

"""
# test calling previously-compiled code again and test re-compilation

x1 = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len))
x2 = torch.randint(0, llama_config.vocab_size, (batch_size * 2, seq_len))

model(x1)
model(x2)
compiled_model(x1)
compiled_model(x2)
compiled_model(x1)
compiled_model(x2)
"""

"""
# time training throughput

import time

optim = optim.Adam(model.parameters())


# get actors for backwards pass
stg1 = compiled_model._ray_actors[1]
stg2 = compiled_model._ray_actors[2]

# warmup
for i in range(2):
    # warmup original model fwd/bwd
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optim.step()
    optim.zero_grad()

    # warmup compiled model fwd/bwd
    out = compiled_model(x)
    grad2 = stg2.backward.remote(pred=out, truth=y, loss_fn=loss_fn)
    grad1 = stg1.backward.remote(grad=grad2)
    upd1 = stg1.update.remote(grad1)
    upd2 = stg2.update.remote(grad2)
    ray.get(upd1, upd2)

# timed

iters = 10

start_compiled = time.perf_counter()
for i in range(iters):
    compiled_model(x)
    grad2 = stg2.backward.remote(truth=y, loss_fn=loss_fn)
    grad1 = stg1.backward.remote(grad=grad2)
    upd1 = stg1.update.remote(grad1)
    upd2 = stg2.update.remote(grad2)
    ray.get(upd1, upd2)
end_compiled = time.perf_counter()

start_original = time.perf_counter()
for i in range(iters):
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optim.step()
    optim.zero_grad()
end_original = time.perf_counter()

print(f"Compiled time: {(end_compiled-start_compiled)*1000/iters:.2f}ms")
print(f"Original time: {(end_original-start_original)*1000/iters:.2f}ms")
# """
