import torch
from torch import nn, optim
from llama import Transformer, LLAMA_DEBUG, LLAMA_1B

import torch._dynamo

llama_config = LLAMA_DEBUG

model = Transformer(llama_config)
compiled_model = torch.compile(model, distribute=True)

batch_size = 100
seq_len = 32
x = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len))
y = torch.zeros((batch_size, llama_config.vocab_size), dtype=torch.long)

"""
# TODO: test back propagation
out1 = model(x)
out2 = compiled_model(x)

loss1 = torch.nn.CrossEntropyLoss()(out1, y)
loss2 = torch.nn.CrossEntropyLoss()(out2, y)

loss1.backward()
loss2.backward()

out11 = model(x)
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
# """

# """
# time overheads
import time

# warmup
for i in range(5):
    compiled_model(x)
    model(x)

# timed

start_compiled = time.perf_counter()
for i in range(5):
    compiled_model(x)
end_compiled = time.perf_counter()

start_original = time.perf_counter()
for i in range(10):
    model(x)
end_original = time.perf_counter()

print(f"Compiled time: {end_compiled-start_compiled:.6f}")
print(f"Original time: {end_original-start_original:.6f}")
# """