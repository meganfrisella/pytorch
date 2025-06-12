import ray
import torch
from torch import nn

from models.stage_manager import stage_manager

class MicroModel(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.c = torch.ones((out_dim))
        self.layers = nn.ModuleList([
            nn.Linear(inp_dim, out_dim),
            nn.Linear(inp_dim, out_dim)])
        self.norm = nn.RMSNorm((out_dim))

    def forward(self, x, y):
        torch._dynamo.distributed_stage(1, optim=torch.optim.Adam)
        x = self.layers[0](x)

        torch._dynamo.distributed_stage(2, optim=torch.optim.Adam)
        y = self.layers[1](y)
        y = self.norm(y)

        torch._dynamo.distributed_stage(3, optim=torch.optim.Adam)
        z = x + y + self.c
        return z


inp_dim = 32
out_dim = 1
model = MicroModel(inp_dim, out_dim)
compiled_model = torch.compile(model, distribute=True)


import time

batch_size = 10
x = torch.randn((batch_size, inp_dim))
y = torch.randn((batch_size, inp_dim))

out = model(x, y)
print(out)

out = compiled_model(x, y)
print(out.get())