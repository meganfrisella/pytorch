import ray
import torch
from torch import nn, optim
from models.llama import Transformer, LLAMA_DEBUG, LLAMA_1B

llama_config = LLAMA_DEBUG

loss_fn = torch.nn.CrossEntropyLoss()
optim_fn = optim.Adam

model = Transformer(llama_config)
compiled_model = torch.compile(model, distribute=True)
compiled_model._set_optimizer(optim_fn)

batch_size = 100
seq_len = 32
x = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len))
y = torch.zeros((batch_size, llama_config.vocab_size), dtype=torch.long)

"""
# test backprop
optim = optim_fn(model.parameters())

out1 = model(x)
out2 = compiled_model(x)

loss = loss_fn(out1, y)
loss.backward()
optim.step()
optim.zero_grad()

for idx, actor in enumerate(reversed(compiled_model.ray_actors)):
    if idx == 0:
        grad = actor.backward.remote(truth=y, loss_fn=loss_fn)
    else:
        grad = actor.backward.remote(grad=grad)
ray.get(grad)

for actor in compiled_model.ray_actors:
    ray.get(actor.update.remote())

print("original:")
out11 = model(x)
print("compiled:")
out22 = compiled_model(x)
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

# """
# time overheads
import time

# warmup
for i in range(2):
    model(x)
    compiled_model(x)

# print("SLEEPING")
# time.sleep(5)

# # timed

# iters = 10

# start_compiled = time.perf_counter()
# for i in range(iters):
#     compiled_model(x)
# end_compiled = time.perf_counter()

# start_original = time.perf_counter()
# for i in range(iters):
#     model(x)
# end_original = time.perf_counter()

# print(f"Compiled time: {(end_compiled-start_compiled)*1000/iters:.2f}ms")
# print(f"Original time: {(end_original-start_original)*1000/iters:.2f}ms")
# """
