import ray
import torch
from torch import nn

import os
os.environ["RAY_PROFILING"] = str(1)
os.environ["RAY_task_events_report_interval_ms"] = str(0)

ray.init()

class MicroModel(nn.Module):
    def __init__(self):
        import os
        os.environ["RAY_PROFILING"] = str(1)
        os.environ["RAY_task_events_report_interval_ms"] = str(0)
        super().__init__()

    def forward(self, x):
        x = x + 1
        print(x)
        x = x + 1
        return x


model = MicroModel()
compiled_model = torch.compile(model, distribute=True)

import time

x = torch.Tensor([3])

# warmup
for i in range(5):
    compiled_model(x)
    model(x)

# time
start_compiled = time.perf_counter()
for i in range(10):
    compiled_model(x)
end_compiled = time.perf_counter()

start_original = time.perf_counter()
for i in range(10):
    model(x)
end_original = time.perf_counter()

print(f"Compiled time: {(end_compiled-start_compiled)*1000000:.2f}")
print(f"Original time: {(end_original-start_original)*1000000:.2f}")

import time
time.sleep(1)
ray.timeline(filename="timeline.json")