import ray
import torch
import logging

import os
import time
import uuid

import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

_fake_tensor_mode = FakeTensorMode()
_fake_tensor_converter = _fake_tensor_mode.fake_tensor_converter


class RemoteTensorKey:
    def __init__(self):
        self.key = str(uuid.uuid4())


def get_fake_tensors(example_inputs: list[torch.Tensor], graph_module: fx.GraphModule):
    fake_inputs = []
    for inp in example_inputs:
        fake_inputs.append(_fake_tensor_converter.from_real_tensor(_fake_tensor_mode, inp))
    with _fake_tensor_mode:
        fakes = graph_module.forward(*fake_inputs)
    return fakes


class RemoteTensor(torch.Tensor):
    _fake: torch.Tensor

    def __new__(cls, 
                fake: list[FakeTensor], 
                obj_ref: ray._raylet.ObjectRef):
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
        instance.resolved = None
        instance._fake = fake
        instance.key = RemoteTensorKey()
        return instance

    def get(self):
        if self.resolved is None:
            obj = ray.get(self.obj_ref)
            if isinstance(obj, list):
                assert len(obj) == 1
                self.resolved = obj[0]
            else:
                self.resolved = obj
        return self.resolved
    
    def get_ref(self):
        if self.resolved is None:
            return self.obj_ref
        else:
            return None

    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(x):
            return x.get() if isinstance(x, RemoteTensor) else x
        print(f"dispatch {func} on {args}")
        args = list(map(unwrap, args))
        kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        out = func(*args, **kwargs)
        return out
    
    def __repr__(self):
        return f"RayTensor(obj_ref={self.obj_ref})"


@ray.remote
class StageActor:
    # def __init__(self, id, compiler_fn, example_inputs, parameters, optim_fn=None):
    def __init__(self, id, optim_fn=None):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

        self.log.info(f"Initializing Ray actor {id} with PID: {os.getpid()}")

        start = time.perf_counter()

        self.actor_id = id
        self.optim_fn = optim_fn

        # map compile id -> compiled fx.Graph function
        self.compiled_fns = dict()
        # map compile id -> model parameters used by the fx.Graph
        self.parameters = dict()
        # map compile id -> optimizer for the fx.Graph
        self.optims = dict()

        end = time.perf_counter()
        self.log.debug(f"__init__ took {(end-start)*1000:.2f}ms")

    def id(self):
        return self.actor_id
    
    def compile_graph(self, id, gm_data, compiler_fn, example_inputs, parameters):
        start = time.perf_counter()

        self.gm = torch.load(gm_data, weights_only=False)
        compiled_fn = compiler_fn(self.gm, example_inputs)
        assert callable(compiled_fn), "compiler_fn did not return callable"
        self.compiled_fns[id] = compiled_fn
        self.parameters[id] = parameters
        if parameters:
            assert self.optim_fn
            self.optims[id] = self.optim_fn(
                [p for p in parameters if p is not None])

        end = time.perf_counter()
        self.log.debug(f"compile_graph took {(end-start)*1000:.2f}ms")
        return "Finished compiling"

    def call(self, id, *args):
        self.log.info(f"Calling forward on actor {self.actor_id} with {len(args)} args")
        start = time.perf_counter()

        start_args = time.perf_counter()

        # Ray object refs resolve to a single element list
        def unwrap(x):
            if isinstance(x, list):
                assert len(x) == 1
            return x[0] if isinstance(x, list) else x
        args = list(map(unwrap, args))

        # TODO: assume the first input is the one we will backprop on
        self.prev_activation = args[0]
        if torch.is_floating_point(self.prev_activation):
            self.prev_activation.requires_grad_()

        new_args = list(args)
        args_plus_parameters = []
        if id in self.parameters:
            parameters = self.parameters[id]
            for arg in parameters:
                if arg is not None:
                    args_plus_parameters.append(arg)
                else:
                    args_plus_parameters.append(new_args[0])
                    del new_args[0]
        else:
            args_plus_parameters = new_args

        end_args = time.perf_counter()

        out = self.compiled_fns[id](*args_plus_parameters)

        act = [t for t in out if t.requires_grad]
        if act:
            assert len(act) == 1
            self.activation = act[0]
        else: 
            self.activation = None

        end = time.perf_counter()
        self.log.debug(f"forward took {(end-start)*1000:.2f}ms")
        self.log.debug(f"processing args took {(end_args-start_args)*1000:.2f}ms")
        return out

    def backward(self, grad=None, truth=None, loss_fn=None):
        assert self.activation
        if loss_fn:
            assert truth is not None
            loss = loss_fn(self.activation, truth)
            loss.backward()
        else:
            assert grad is not None
            self.activation.backward(grad)

        if self.prev_activation.requires_grad:
            return self.prev_activation.grad
        else:
            return None

    def update(self):
        assert self.optim_fn
        self.optim.step()
        self.optim.zero_grad()
        return "done"
