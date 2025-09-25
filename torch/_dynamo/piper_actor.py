import ray
import torch
import logging

import os
import time
import uuid

import types
import inspect

import json, importlib, inspect, operator, builtins
import torch.fx as fx
import json

from collections import defaultdict

from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch._guards import CompileId

_fake_tensor_mode = FakeTensorMode()
_fake_tensor_converter = _fake_tensor_mode.fake_tensor_converter

torch.set_float32_matmul_precision('high')


# ---------- encoders/decoders for args ----------
def encode_arg(a):
    if isinstance(a, fx.Node):
        return {"__node__": a.name}
    if isinstance(a, torch.device):
        return {"__device__": str(a)}
    if isinstance(a, torch.dtype):
        return {"__dtype__": str(a).replace("torch.", "")}
    if isinstance(a, slice):
        return {"__slice__": True,
                "start": encode_arg(a.start),
                "stop": encode_arg(a.stop),
                "step": encode_arg(a.step)}
    if a is Ellipsis:
        return {"__ellipsis__": True}
    if isinstance(a, tuple):  # <-- preserve tuples
        return {"__tuple__": [encode_arg(x) for x in a]}
    if isinstance(a, list):
        return [encode_arg(x) for x in a]
    if isinstance(a, dict):
        return {k: encode_arg(v) for k, v in a.items()}
    return a

def decode_arg(a, name_to_node):
    if isinstance(a, dict):
        if "__node__" in a:
            return name_to_node[a["__node__"]]
        if "__device__" in a:
            return torch.device(a["__device__"])
        if "__dtype__" in a:
            return getattr(torch, a["__dtype__"])
        if "__slice__" in a:
            return slice(
                decode_arg(a["start"], name_to_node),
                decode_arg(a["stop"], name_to_node),
                decode_arg(a["step"], name_to_node),
            )
        if "__ellipsis__" in a:
            return Ellipsis
        if "__tuple__" in a:  # <-- reconstruct tuples
            return tuple(decode_arg(x, name_to_node) for x in a["__tuple__"])
        # generic dict
        return {k: decode_arg(v, name_to_node) for k, v in a.items()}
    if isinstance(a, list):
        return [decode_arg(x, name_to_node) for x in a]
    return a

# ---------- target (callable/op) serializer ----------
def _is_op_overload(obj):
    # Works across PyTorch versions without importing private types directly
    return obj.__class__.__module__.startswith("torch._ops") or obj.__class__.__name__.startswith("OpOverload")

def serialize_target(t):
    # print("SERIALIZING", t)
    # call_method uses a string method name, pass through
    if isinstance(t, str):
        return {"kind": "string", "value": t}

    # Handle _VariableFunctionsClass
    if getattr(t, "__module__", "") == "torch._VariableFunctionsClass":
        public_name = t.__name__
        if hasattr(torch, public_name):
            return {"kind": "py_func", "module": "torch", "qualname": public_name}
        else:
            raise ValueError(f"No public torch alias for {t}")

    # torch.ops.* (aten, prim, etc.)
    if _is_op_overload(t) or (getattr(t, "__module__", "").startswith("torch._ops")):
        return {"kind": "torch_op", "path": str(t)}  # e.g. "aten.add.Tensor" or "aten.add"

    # regular python function or built-in
    if inspect.isfunction(t) or inspect.isbuiltin(t):
        mod = inspect.getmodule(t)
        if mod is None:
            raise ValueError(f"Cannot serialize function without module: {t}")
        return {"kind": "py_func", "module": mod.__name__, "qualname": t.__name__}

    # classes or callables rarely appear as call_function targets, but support anyway
    if inspect.isclass(t):
        mod = t.__module__
        return {"kind": "py_obj", "module": mod, "qualname": t.__qualname__}

    # operator functions (already covered by py_func, but ensure resolvable)
    if t in operator.__dict__.values():
        return {"kind": "py_func", "module": "operator", "qualname": t.__name__}

    # last resort: try module+name
    mod = getattr(t, "__module__", None)
    name = getattr(t, "__name__", None)
    if mod and name:
        return {"kind": "py_func", "module": mod, "qualname": name}

    raise NotImplementedError(f"Unsupported target type: {t} ({type(t)})")

def _resolve_qualname(mod, qualname):
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj

def deserialize_target(payload):
    # print("DESERIALIZING", payload)
    kind = payload["kind"]

    if kind == "string":
        return payload["value"]

    if kind == "py_func":
        mod = importlib.import_module(payload["module"])
        return _resolve_qualname(mod, payload["qualname"])

    if kind == "py_obj":
        mod = importlib.import_module(payload["module"])
        return _resolve_qualname(mod, payload["qualname"])

    if kind == "torch_op":
        # payload["path"] like "aten.add.Tensor" or "aten.add"
        obj = torch.ops
        for part in payload["path"].split("."):
            obj = getattr(obj, part)
        return obj

    raise NotImplementedError(f"Unknown target kind: {kind}")

# ---------- main I/O ----------
def serialize_graphmodule(gm: fx.GraphModule) -> str:
    nodes = []
    for n in gm.graph.nodes:
        nodes.append({
            "name": n.name,
            "op": n.op,
            "target": serialize_target(n.target) if n.op in ("call_function", "call_method", "call_module", "get_attr") else None,
            "args": encode_arg(n.args),
            "kwargs": encode_arg(n.kwargs),
        })

    data = {
        "nodes": nodes,
        "state_dict": {k: v.detach().cpu().tolist() for k, v in gm.state_dict().items()},
        # save which device parameters were on, optional:
        "param_devices": {k: str(v.device) for k, v in gm.state_dict().items()},
    }
    serialized = json.dumps(data, ensure_ascii=False)

    # MEMORY CLEANUP
    del data
    del nodes

    return serialized

def _unwrap_output_arg(decoded):
    # FX stores output as (value,), where value may itself be a tuple.
    # Inductor expects the inner tuple directly.
    if isinstance(decoded, (tuple, list)) and len(decoded) == 1 and isinstance(decoded[0], (tuple, list)):
        return decoded[0]
    return decoded

def deserialize_graphmodule(s: str) -> fx.GraphModule:
    data = json.loads(s)
    g = fx.Graph()
    name_to_node = {}

    for n in data["nodes"]:
        op = n["op"]
        if op == "placeholder":
            node = g.placeholder(n["name"])
        elif op == "output":
            decoded = decode_arg(n["args"], name_to_node)
            node = g.output(_unwrap_output_arg(decoded))   # <-- unwrap here
        elif op == "call_function":
            target = deserialize_target(n["target"])
            args = decode_arg(n["args"], name_to_node)
            kwargs = decode_arg(n["kwargs"], name_to_node)
            node = g.call_function(target, tuple(args), kwargs)
        elif op == "call_method":
            target = deserialize_target(n["target"])
            args = decode_arg(n["args"], name_to_node)
            kwargs = decode_arg(n["kwargs"], name_to_node)
            node = g.call_method(target, tuple(args), kwargs)
        elif op == "call_module":
            target = deserialize_target(n["target"])
            args = decode_arg(n["args"], name_to_node)
            kwargs = decode_arg(n["kwargs"], name_to_node)
            node = g.call_module(target, tuple(args), kwargs)
        elif op == "get_attr":
            target = deserialize_target(n["target"])
            node = g.get_attr(target)
        else:
            raise NotImplementedError(f"op {op} not handled")
        name_to_node[n["name"]] = node

    gm = fx.GraphModule(torch.nn.Module(), g)
    state = {k: torch.tensor(v) for k, v in data["state_dict"].items()}
    gm.load_state_dict(state, strict=False)
    return gm


class RemoteTensorKey:
    def __init__(self):
        self.key = str(uuid.uuid4())



class RemoteTensor(torch.Tensor):
    _fake: torch.Tensor
    _stage_id: int

    def __new__(cls, 
                fake: FakeTensor, 
                obj_ref: ray._raylet.ObjectRef,
                stage_id: int):
        instance = torch.Tensor._make_wrapper_subclass(
            cls,
            fake.size(),
            strides=fake.stride(),
            storage_offset=fake.storage_offset(),
            device=fake.device,  # This is the device of of either input tensor or first tensor of a list
            dtype=fake.dtype,
            layout=fake.layout,
            requires_grad=fake.requires_grad,
        )
        instance.obj_ref = obj_ref
        instance._stage_id = stage_id
        instance.resolved = None
        instance._fake = fake
        instance.key = RemoteTensorKey()
        return instance

    def get_stage_id(self):
        return self._stage_id

    def get(self):
        if self.resolved is None:
            obj = ray.get(self.obj_ref)
            if isinstance(obj, list) or isinstance(obj, tuple):
                assert len(obj) == 1
                self.resolved = obj[0]
            else:
                self.resolved = obj
        return self.resolved
    
    def get_ref(self):
        return self.obj_ref

    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if torch._dynamo.eval_frame.dynamo_tls.currently_compiling:
            # print("HERE 1", func, args)
            # fall back to fake execution to keep tracing alive
            def unwrap_fake(x):
                if isinstance(x, RemoteTensor):
                    return x._fake
                return x
            args = torch.utils._pytree.tree_map(unwrap_fake, args)
            kwargs = torch.utils._pytree.tree_map(unwrap_fake, kwargs or {})
            return func(*args, **kwargs)
        
        # print("HERE 2", func, args)
        def unwrap(x):
            if isinstance(x, RemoteTensor):
                return x.get()
            elif isinstance(x, (list, tuple)):
                return type(x)(unwrap(v) for v in x)
            elif isinstance(x, dict):
                return {k: unwrap(v) for k, v in x.items()}
            else:
                return x

        args = torch.utils._pytree.tree_map(unwrap, args)
        kwargs = torch.utils._pytree.tree_map(unwrap, kwargs or {})

        out = func(*args, **kwargs)
        return out

    # arithmetic
    def __add__(self, other):
        return self.get() + (other.get() if isinstance(other, RemoteTensor) else other)

    def __radd__(self, other):
        return (other.get() if isinstance(other, RemoteTensor) else other) + self.get()

    def __sub__(self, other):
        return self.get() - (other.get() if isinstance(other, RemoteTensor) else other)

    def __rsub__(self, other):
        return (other.get() if isinstance(other, RemoteTensor) else other) - self.get()

    # comparisons
    def __eq__(self, other):
        return self.get() == (other.get() if isinstance(other, RemoteTensor) else other)

    def __lt__(self, other):
        return self.get() < (other.get() if isinstance(other, RemoteTensor) else other)

    def __le__(self, other):
        return self.get() <= (other.get() if isinstance(other, RemoteTensor) else other)

    def __gt__(self, other):
        return self.get() > (other.get() if isinstance(other, RemoteTensor) else other)

    def __ge__(self, other):
        return self.get() >= (other.get() if isinstance(other, RemoteTensor) else other)

    def __repr__(self):
        return f"RemoteTensor(obj_ref={self.obj_ref})"

@ray.remote
class StageActor:
    # def __init__(self, id, compiler_fn, example_inputs, parameters, optim_fn=None):
    def __init__(self, id, optim_fn=None):
        torch.manual_seed(0)

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

        self.log.info(f"Initializing Ray actor {id} with PID: {os.getpid()}")

        start = time.perf_counter()

        self.actor_id = id
        self.optim_fn = optim_fn

        self.input = None
        self.truth = None
        self.fwd_objs = {}
        self.bwd_objs = {}

        # ordered list of frame ids for ordering the fx.Graphs on this actor
        self.frame_ids = []
        # map compile id -> compiled fx.Graph function
        self.compiled_fns = dict()
        # map compile id -> model parameters used by the fx.Graph
        self.parameters = dict()
        # map compile id -> optimizer for the fx.Graph
        self.optims = dict()
        # map stage_id -> mb_idx -> previous activation (if this stage is not first)
        self.prev_activations = defaultdict(dict)
        # map stage id -> mb_idx -> current activation
        self.activations = defaultdict(dict)
        # accumuate loss for each microbatch
        self.loss = []

        # Timing infrastructure
        self.tracing = False  # Toggle for timing and memory tracing
        self.trace_data = {'update': {'total': [], 'peak_memory_delta': [], 'peak_memory': []}}

        end = time.perf_counter()
        self.log.debug(f"__init__ took {(end-start)*1000:.2f}ms")

    def id(self):
        return self.actor_id

    def send_input(self, tensor):
        self.input = tensor.to('cuda')
        return "done"
    
    def send_truth(self, tensor):
        self.truth = tensor.to('cuda')
        return "done"

    def compile_graph(self, compile_id: CompileId, stage_id, gm_data, compiler_fn, example_inputs, parameters):
        self.log.info(f"Compiling graph on actor {self.actor_id}. compile id: {compile_id}. inputs: {len(example_inputs)}")
        start = time.perf_counter()

        self.trace_data[stage_id] = {
            'forward': {
                'pre_forward': [],
                'forward': [],
                'post_forward': [],
                'peak_memory_delta': [],
                'peak_memory': []
            },
            'backward': {
                'pre_backward': [],
                'backward': [],
                'post_backward': [],
                'peak_memory_delta': [],
                'peak_memory': []
            },
        }

        # if this is a recompile, assert the non-null parameters have the same shape
        # TODO: this is a weak check
        frame_id = compile_id.frame_id
        if frame_id in self.parameters:
            old_params = self.parameters[frame_id]
            non_null_old_params = [p for p in old_params if p is not None]
            non_null_new_params = [p for p in parameters if p is not None]

            # assert len(non_null_old_params) == len(non_null_new_params)
            # for p1, p2 in zip(non_null_old_params, non_null_new_params):
            #     assert p1.shape == p2.shape
            # self.log.info(f"Recompiling frame_id {frame_id} on actor {self.actor_id}")

            # save the new null pattern with the original parameters values
            # if len(old_params) != len(parameters):
            #     old_idx = 0
            #     for new_idx, p in enumerate(parameters):
            #         if p is not None:
            #             parameters[new_idx] = non_null_old_params[old_idx]
            #             old_idx += 1
            self.parameters[frame_id] = parameters

        # otherwise if this is a fresh compile, save the parameters and initialize the optimizer
        else:
            def send_to_device(param):
                if isinstance(param, torch.Tensor):
                    return param.to('cuda').detach().requires_grad_()
                else:
                    return param
            parameters = list(map(send_to_device, parameters))
            self.parameters[frame_id] = parameters
            non_null_params = [p for p in parameters if p is not None]
            if non_null_params:
                assert self.optim_fn
                self.optims[frame_id] = self.optim_fn([p for p in non_null_params if isinstance(p, torch.Tensor)])
        
        # save the frame id in an ordered list
        if frame_id not in self.frame_ids:
            self.frame_ids.append(frame_id)

        gm = deserialize_graphmodule(gm_data)
        compiled_fn = compiler_fn(gm, example_inputs)
        assert callable(compiled_fn), "compiler_fn did not return callable"
        # compiled_fn(*example_inputs)
        self.compiled_fns[compile_id] = compiled_fn

        # MEMORY CLEANUP
        del gm
        del gm_data
        del example_inputs      
        import gc
        gc.collect()

        end = time.perf_counter()
        self.log.debug(f"compile_graph took {(end-start)*1000:.2f}ms")
        return "Finished compiling"

    @ray.method(tensor_transport="nccl")
    def call(self, compile_id: CompileId, stage_id: int, mb_idx: int, *args):
        self.log.debug(f"Calling forward {stage_id} mb {mb_idx} on actor {self.actor_id} with {len(args)} args")
        
        # Initialize timing variables
        if self.tracing:
            # Create CUDA events for timing
            start_event = torch.cuda.Event(enable_timing=True)
            pre_forward_event = torch.cuda.Event(enable_timing=True)
            forward_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Reset peak memory counter for forward pass
            torch.cuda.reset_peak_memory_stats()
            forward_start_memory = torch.cuda.memory_allocated()

            # Start timing
            start_event.record()
        
        def pre_loaded_input(param):
            if param is None:
                return self.input
            else:
                return param
        args = list(map(pre_loaded_input, args))

        # Ray object refs resolve to a single element list
        def unwrap(x):
            if isinstance(x, list) or isinstance(x, tuple):
                assert len(x) == 1
            return x[0] if isinstance(x, list) or isinstance(x, tuple) else x
        args = list(map(unwrap, args))

        # def send_to_device(param):
        #     if isinstance(param, torch.Tensor):
        #         return param.to('cuda')
        #     else:
        #         return param
        # args = list(map(send_to_device, args))

        # save all inputs with gradients as previous activations for backprop
        frame_id = compile_id.frame_id
        # if frame_id == self.frame_ids[0]:
        if stage_id != 0:
            activation_idxs = self.prev_activation_idxs
            prev_activations = []
            for idx, arg in enumerate(args):
                if idx in activation_idxs:
                    # print(f"prev stage {stage_id} idx {idx}", type(arg))
                    prev_activations.append(arg.requires_grad_())
            self.prev_activations[stage_id][mb_idx] = prev_activations

        # patch the args into the stored parameters list, which has None values for the args
        frame_id = compile_id.frame_id
        new_args = list(args)
        parameters = self.parameters[frame_id]
        args_plus_parameters = []

        assert len(new_args) == len([p for p in parameters if p is None])
        if frame_id in self.parameters:
            for arg in self.parameters[frame_id]:
                if arg is not None:
                    args_plus_parameters.append(arg)
                else:
                    args_plus_parameters.append(new_args[0])
                    del new_args[0]
        else:
            args_plus_parameters = new_args

        # Record timing before forward call
        if self.tracing:
            pre_forward_event.record()

        out = self.compiled_fns[compile_id](*args_plus_parameters)

        # Record timing after forward call
        if self.tracing:
            forward_event.record()

        del args_plus_parameters
        del args
        del parameters
        del new_args

        # save all outputs which require gradients as the activations
        # if frame_id == self.frame_ids[-1]:
        # for idx, t in enumerate(out):
        #     print(f"STAGE {stage_id} MB {mb_idx} ACT {idx}", t.requires_grad)
        self.activations[stage_id][mb_idx] = [t for t in out if isinstance(t, torch.Tensor) and t.requires_grad]

        # End timing and record data
        if self.tracing:
            end_event.record()
            
            # Synchronize and record timing data
            torch.cuda.synchronize()
            pre_forward_time = start_event.elapsed_time(pre_forward_event)
            forward_time = pre_forward_event.elapsed_time(forward_event)
            post_forward_time = forward_event.elapsed_time(end_event)
            
            forward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - forward_start_memory) / (1024**3)
            forward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

            self.trace_data[stage_id]['forward']['pre_forward'].append(pre_forward_time)
            self.trace_data[stage_id]['forward']['forward'].append(forward_time)
            self.trace_data[stage_id]['forward']['post_forward'].append(post_forward_time)
            self.trace_data[stage_id]['forward']['peak_memory_delta'].append(forward_peak_memory_delta_gb)
            self.trace_data[stage_id]['forward']['peak_memory'].append(forward_peak_memory_gb)

            if isinstance(out, list) or isinstance(out, tuple):
                self.fwd_objs[stage_id] = [torch.ones_like(t) for t in out]
            else:
                self.fwd_objs[stage_id] = torch.ones_like(out)
        return out

    def call_cpu(self, compile_id: CompileId, stage_id: int, mb_idx: int, *args):
        self.log.debug(f"Calling cpu forward {stage_id} mb {mb_idx} on actor {self.actor_id} with {len(args)} args")

        def pre_loaded_input(param):
            if param is None:
                return self.input
            else:
                return param
        args = list(map(pre_loaded_input, args))

        # Ray object refs resolve to a single element list
        def unwrap(x):
            if isinstance(x, list) or isinstance(x, tuple):
                assert len(x) == 1
                return x[0]
            else:
                return x
        args = list(map(unwrap, args))

        def send_to_device(param):
            if isinstance(param, torch.Tensor):
                return param.to('cuda')
            else:
                return param
        args = list(map(send_to_device, args))

        # save all inputs with gradients as previous activations for backprop
        frame_id = compile_id.frame_id
        # if frame_id == self.frame_ids[0]:
        if stage_id != 0:
            activation_idxs = []
            prev_activations = []
            for idx, arg in enumerate(args):
                if isinstance(arg, torch.Tensor) and arg.requires_grad:
                    # print(f"prev stage {stage_id} idx {idx}", type(arg))
                    prev_activations.append(arg)
                    activation_idxs.append(idx)
            self.prev_activations[stage_id][mb_idx] = prev_activations
            self.prev_activation_idxs = activation_idxs

        # patch the args into the stored parameters list, which has None values for the args
        frame_id = compile_id.frame_id
        new_args = list(args)
        parameters = self.parameters[frame_id]
        args_plus_parameters = []

        assert len(new_args) == len([p for p in parameters if p is None])
        if frame_id in self.parameters:
            for arg in self.parameters[frame_id]:
                if arg is not None:
                    args_plus_parameters.append(arg)
                else:
                    args_plus_parameters.append(new_args[0])
                    del new_args[0]
        else:
            args_plus_parameters = new_args


        out = self.compiled_fns[compile_id](*args_plus_parameters)

        del args_plus_parameters
        del args
        del parameters
        del new_args

        # save all outputs which require gradients as the activations
        # if frame_id == self.frame_ids[-1]:
        # for idx, t in enumerate(out):
            # print(f"act stage {stage_id} idx {idx}", t.requires_grad)
        self.activations[stage_id][mb_idx] = [t for t in out if isinstance(t, torch.Tensor) and t.requires_grad]
        # print(f"output activations stage {stage_id}:", len(self.activations[stage_id][mb_idx]))
        return out

    @ray.method(tensor_transport="nccl")
    def backward(self, stage_id: int, mb_idx: int, inp, loss_fn=None):
        self.log.debug(f"Calling backward {stage_id} mb {mb_idx} on actor {self.actor_id}", inp)
        
        # Initialize timing variables
        if self.tracing:
            # Create CUDA events for timing
            start_event = torch.cuda.Event(enable_timing=True)
            pre_backward_event = torch.cuda.Event(enable_timing=True)
            backward_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Start timing
            start_event.record()
        
        activation = self.activations[stage_id][mb_idx]
        assert activation
        activation = activation[0]
        
        # Reset peak memory counter for backward pass
        if self.tracing:
            torch.cuda.reset_peak_memory_stats()
            backward_start_memory = torch.cuda.memory_allocated()

        # Record timing before loss/backward call
        if self.tracing:
            pre_backward_event.record()
        
        # compute loss in the last stage. use the saved activation rather
        # than inp because the saved activation remembers the computation graph
        if loss_fn is not None:
            if self.truth is not None:
                labels = self.truth
            else:
                labels = inp[0]
            loss = loss_fn(activation, labels)
            self.loss.append(loss)
            loss.backward()
        # if not the last stage, backprop on the stored activation given 
        # the input gradient from the subsequent stage
        else:
            assert inp is not None
            assert activation.shape == inp.shape
            activation.backward(gradient=inp)

        # Record timing after loss/backward call
        if self.tracing:
            backward_event.record()

        del self.activations[stage_id][mb_idx]
        del activation

        if stage_id != 0:
            prev_activations = self.prev_activations[stage_id][mb_idx]
            assert prev_activations
            ret = [act.grad for act in prev_activations]
            del self.prev_activations[stage_id][mb_idx]
            del prev_activations
        else:
            ret = ["done"]
        
        # End timing and record data
        if self.tracing:
            end_event.record()
            
            # Synchronize and record timing data
            torch.cuda.synchronize()
            pre_backward_time = start_event.elapsed_time(pre_backward_event)
            backward_time = pre_backward_event.elapsed_time(backward_event)
            post_backward_time = backward_event.elapsed_time(end_event)
            backward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - backward_start_memory) / (1024**3)
            backward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

            self.trace_data[stage_id]['backward']['pre_backward'].append(pre_backward_time)
            self.trace_data[stage_id]['backward']['backward'].append(backward_time)
            self.trace_data[stage_id]['backward']['post_backward'].append(post_backward_time)
            self.trace_data[stage_id]['backward']['peak_memory_delta'].append(backward_peak_memory_delta_gb)
            self.trace_data[stage_id]['backward']['peak_memory'].append(backward_peak_memory_gb)
            
            if stage_id != 0:
                self.bwd_objs[stage_id] = torch.ones_like(ret[0])
        return ret + ret

    def update(self, *done_mbs):
        self.log.debug(f"Calling update on actor {self.actor_id}")
        
        # Initialize timing variables
        if self.tracing:
            # Create CUDA events for timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Reset peak memory counter for update
            torch.cuda.reset_peak_memory_stats()
            update_start_memory = torch.cuda.memory_allocated()

            # Start timing
            start_event.record()
        
        assert self.optim_fn
        for _, optim in self.optims.items():
            optim.step()
            optim.zero_grad()
        losses = self.loss
        self.loss.clear()
        
        # End timing and record data
        if self.tracing:
            end_event.record()
            
            # Synchronize and record timing data
            torch.cuda.synchronize()
            total_time = start_event.elapsed_time(end_event)
            update_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - update_start_memory) / (1024**3)
            update_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

            self.trace_data['update']['total'].append(total_time)
            self.trace_data['update']['peak_memory_delta'].append(update_peak_memory_delta_gb)
            self.trace_data['update']['peak_memory'].append(update_peak_memory_gb)

        return losses

    def get_trace_data(self) -> dict:
        """
        Retrieve timing data collected during training.
        
        Returns:
            dict: Dictionary containing timing data for call, backward, and update functions.
                 Each function has sub-dictionaries with timing measurements in milliseconds.
        """
        return self.trace_data.copy()
    
    def clear_trace_data(self) -> None:
        """
        Clear all collected timing data.
        """
        for stage_id in self.trace_data:
            self.trace_data[stage_id] = {
                'forward': {
                    'pre_forward': [],
                    'forward': [],
                    'post_forward': [],
                    'peak_memory_delta': [],
                    'peak_memory': []
                },
                'backward': {
                    'pre_backward': [],
                    'backward': [],
                    'post_backward': [],
                    'peak_memory_delta': [],
                    'peak_memory': []
                },
            }
        self.trace_data['update'] = {
            'total': [],
            'peak_memory_delta': [],
            'peak_memory': []
        }

    def set_tracing(self, enabled: bool) -> None:
        """
        Enable or disable timing and memory tracing.
        
        Args:
            enabled (bool): True to enable tracing, False to disable.
        """
        self.tracing = enabled
        self.log.info(f"Actor {self.actor_id}: Tracing {'enabled' if enabled else 'disabled'}")

    def start_mem_tracing(self) -> None:
        torch.cuda.memory._record_memory_history()
        return "done"
    
    def stop_mem_tracing(self) -> None:
        torch.cuda.memory._dump_snapshot(f"actor{self.actor_id}_memory_snapshot_mb4_gpipe.pickle")
        print(f"Saved memory snapshot to actor{self.actor_id}_memory_snapshot_mb4_gpipe.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        return "done"
    
    @ray.method(tensor_transport="nccl")
    def get_object(self, is_fwd, stage_id):
        if is_fwd:
            return self.fwd_objs[stage_id]
        else:
            return self.bwd_objs[stage_id]

    def time_object_retrieval(self, obj) -> dict:
        objs = []
        if isinstance(obj, tuple) or isinstance(obj, list):
            objs = [t + 1 for t in obj]
        else:
            objs = [obj + 1]
        self.objs = objs
        return "done"