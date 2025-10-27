import torch
from torch import nn

import ray
ray.init(include_dashboard=True, namespace="mymodel")

# --- Example model ---
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        piper.pipeline_stage(stage_id=0, actor_id=0)
        x = self.relu(self.fc1(x))
        print(x)
        piper.pipeline_stage(stage_id=1, actor_id=1)
        x = self.relu(self.fc2(x))
        piper.pipeline_stage(stage_id=2, actor_id=2)
        x = self.relu(self.fc3(x))
        return x

model = MyModel().to('cuda')
model = torch.compile(model, distribute=True)

x = torch.randn(10, 1024).to('cuda')
y = torch.randn(10, 10).to('cuda')
out = model(x, dynamo_mb=42).get()
print("mb=42", out[0][0])

import dis, types, inspect

resume_fns = [key for key in globals().keys() if "__resume_at" in key]
name_map = {"forward":"forward"}
for key in resume_fns:
    fn = globals()[key]
    name_map[fn.__qualname__] = key

output_codes = torch._dynamo.convert_frame.output_codes
output_fns = {}
for code in output_codes.seen:
    code = code()
    fn = types.FunctionType(code, globals())
    name = name_map[fn.__name__]
    fn.__name__ = name
    output_fns[name] = fn
    globals()[name] = fn

forward = output_fns["forward"]
out = forward(model, x, dynamo_mb=999)
print("mb=999", out[0][0])

import dis, functools
from dataclasses import dataclass
from typing import Optional, Callable
from bytecode import Bytecode, Instr, Label

@dataclass(frozen=True)
class CallMarker:
    next_call: str
    args: tuple
    kwargs: dict

def _make_stub(name: str) -> Callable:
    def _stub(*args, **kwargs):
        return CallMarker(name, args, kwargs)
    _stub.__name__ = f"__stub_{name}"
    return _stub

def _find_resume_name(fn: Callable) -> Optional[str]:
    instrs = list(dis.get_instructions(fn))
    for i, ins in enumerate(instrs):
        if ins.opname.startswith("CALL"):
            # search backward a short window for a LOAD_GLOBAL/LOAD_NAME callee
            for back in range(i - 1, max(i - 12, -1), -1):
                prev = instrs[back]
                if prev.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
                    if isinstance(prev.argval, str) and prev.argval.startswith("__resume_at_"):
                        return prev.argval
                    break
    return None

def _should_split(code: Bytecode) -> bool:
    """
    Return True iff there is a LOAD_ATTR distributed_stage before
    a call to a __resume_at_* global.
    """
    saw_distributed_stage = False
    for instr in code:
        if instr.name == "LOAD_ATTR" and instr.arg == "distributed_stage":
            return True
    return False

def split_before(fn: Callable):
    """
    Wrap `fn`. If `fn` contains a single call to a name starting with "__resume_at_",
    the wrapper replaces that global with a stub that returns CallMarker(args, kwargs).
    The wrapper accepts either:
      - normal calling form: wrapper(*args, **kwargs)
      - chained form: wrapper(marker) where marker is CallMarker
    In chained form the marker's args/kwargs are used as the call to `fn`.
    """
    code = Bytecode.from_code(fn.__code__)
    g = fn.__globals__
    resume_name = _find_resume_name(fn)
    should_split = _should_split(code)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # print("CALLING WRAPPER", fn.__name__)
        # global next_call
        # next_call = resume_name
        # allow chained form: single CallMarker positional argument
        if len(args) == 1 and not kwargs and isinstance(args[0], CallMarker):
            call_args, call_kwargs = args[0].args, args[0].kwargs
        else:
            call_args, call_kwargs = args, kwargs

        # if no resume_name found, just invoke the function (still accept marker)
        if not should_split:
            return fn(*call_args, **call_kwargs)

        assert resume_name is not None
        
        # stub the target resume global, call, then restore
        saved = {}
        if resume_name in g:
            # print("STUBBING", resume_name)
            saved[resume_name] = g[resume_name]
            g[resume_name] = _make_stub(resume_name)
        try:
            return fn(*call_args, **call_kwargs)
        finally:
            # restore original binding if we changed it
            if resume_name in saved:
                g[resume_name] = saved[resume_name]

    return wrapper

transformed_fns = {}
for key, fn in output_fns.items():
    name = fn.__name__
    new_fn = split_before(fn)
    globals()[name] = new_fn
    transformed_fns[name] = new_fn


out = (model, x, 77)
next_call = "forward"
fwd_fns = []
while next_call:
    # print("CALLING OUTER", next_call)
    fn = transformed_fns[next_call]
    fwd_fns.append(fn)
    if isinstance(out, list) or isinstance(out, tuple):
        out = fn(*out)
    else:
        out = fn(out)
    if isinstance(out, CallMarker):
        next_call = out.next_call
    else:
        next_call = None
print("mb=77", out[0][0])

from torch._dynamo.scheduling import Task, DAGEdge, execute_schedule

schedule = [
    [
        Task(stage_id=0, mb_idx=0, is_fwd=True),
        Task(stage_id=0, mb_idx=1, is_fwd=True),
        Task(stage_id=0, mb_idx=2, is_fwd=True),
        Task(stage_id=0, mb_idx=3, is_fwd=True),
        Task(stage_id=2, mb_idx=0, is_fwd=True),
        None,
        Task(stage_id=2, mb_idx=1, is_fwd=True),
        Task(stage_id=2, mb_idx=0, is_fwd=False),
        Task(stage_id=2, mb_idx=2, is_fwd=True),
        Task(stage_id=2, mb_idx=1, is_fwd=False),
        Task(stage_id=2, mb_idx=3, is_fwd=True),
        Task(stage_id=2, mb_idx=2, is_fwd=False),
        None,
        Task(stage_id=2, mb_idx=3, is_fwd=False),
        Task(stage_id=0, mb_idx=0, is_fwd=False),
        Task(stage_id=0, mb_idx=1, is_fwd=False),
        Task(stage_id=0, mb_idx=2, is_fwd=False),
        Task(stage_id=0, mb_idx=3, is_fwd=False),
    ],
    [
        None,
        Task(stage_id=1, mb_idx=0, is_fwd=True),
        Task(stage_id=1, mb_idx=1, is_fwd=True),
        Task(stage_id=1, mb_idx=2, is_fwd=True),
        Task(stage_id=1, mb_idx=3, is_fwd=True),
        Task(stage_id=3, mb_idx=0, is_fwd=True),
        Task(stage_id=3, mb_idx=0, is_fwd=False),
        Task(stage_id=3, mb_idx=1, is_fwd=True),
        Task(stage_id=3, mb_idx=1, is_fwd=False),
        Task(stage_id=3, mb_idx=2, is_fwd=True),
        Task(stage_id=3, mb_idx=2, is_fwd=False),
        Task(stage_id=3, mb_idx=3, is_fwd=True),
        Task(stage_id=3, mb_idx=3, is_fwd=False),
        Task(stage_id=1, mb_idx=0, is_fwd=False),
        Task(stage_id=1, mb_idx=1, is_fwd=False),
        Task(stage_id=1, mb_idx=2, is_fwd=False),
        Task(stage_id=1, mb_idx=3, is_fwd=False),
        None,
    ],
]

schedule = [
    [
        Task(stage_id=0, mb_idx=0, is_fwd=True),
        Task(stage_id=0, mb_idx=1, is_fwd=True),
        None,
        Task(stage_id=0, mb_idx=0, is_fwd=False),
        Task(stage_id=0, mb_idx=2, is_fwd=True),
        Task(stage_id=0, mb_idx=1, is_fwd=False),
        Task(stage_id=0, mb_idx=3, is_fwd=True),
        Task(stage_id=0, mb_idx=2, is_fwd=False),
        None,
        Task(stage_id=0, mb_idx=3, is_fwd=False),
    ],
    [
        None,
        Task(stage_id=1, mb_idx=0, is_fwd=True),
        Task(stage_id=1, mb_idx=0, is_fwd=False),
        Task(stage_id=1, mb_idx=1, is_fwd=True),
        Task(stage_id=1, mb_idx=1, is_fwd=False),
        Task(stage_id=1, mb_idx=2, is_fwd=True),
        Task(stage_id=1, mb_idx=2, is_fwd=False),
        Task(stage_id=1, mb_idx=3, is_fwd=True),
        Task(stage_id=1, mb_idx=3, is_fwd=False),
        None,
    ],
]

loss_fn = torch.nn.CrossEntropyLoss()

for i in range(5):
    out = execute_schedule(model, schedule, [x], y, loss_fn, fwd_fns=fwd_fns)
    ray.get(out)

import time
time.sleep(1)

for i in range(5):
    out = execute_schedule(model, schedule, [x], y, loss_fn, fwd_fns=fwd_fns)
    ray.get(out)

ray.timeline("timeline.json")