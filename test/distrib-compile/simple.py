import ray
import torch
from torch import nn

import os
os.environ["RAY_PROFILING"] = str(1)
os.environ["RAY_task_events_report_interval_ms"] = str(0)

@ray.remote
class Actor:
    def echo(self, x):
        return x

actor = Actor.remote()

ray.get(actor.echo.remote(5))

import time

time.sleep(10)
ray.timeline(filename="timeline.json")