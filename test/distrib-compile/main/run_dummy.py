import ray
import torch
from torch import nn

from models.stage_manager import stage_manager

class MicroModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        torch._dynamo.distributed_stage(1)
        if x < 0:
            x = x - 1
        else:
            x = x + 1

        torch._dynamo.distributed_stage(2)
        y = y + 1

        torch._dynamo.distributed_stage(3)
        z = x + y
        return z


model = MicroModel()
compiled_model = torch.compile(model, distribute=True)
compiled_model._set_stage_dependencies([(1,3), (2,3)])

import time

x = torch.Tensor([1])
y = torch.Tensor([2])

out = compiled_model(x, y)
print("result:", out)