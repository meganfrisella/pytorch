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

        self.id = id
        self.compiled_fns = dict()
        
        # self.compiler_fn = compiler_fn
        # self.example_inputs = example_inputs
        # self.parameters = parameters
        # if parameters and optim_fn:
        #     self.optim = optim_fn([p for p in parameters if p is not None])
        # else:
        #     self.optim = None

        end = time.perf_counter()
        self.log.debug(f"__init__ took {(end-start)*1000:.2f}ms")

    def id(self):
        return self.id
    
    def compile_graph(self, id, gm_data, compiler_fn, example_inputs):
        start = time.perf_counter()

        self.gm = torch.load(gm_data, weights_only=False)
        compiled_fn = compiler_fn(self.gm, example_inputs)
        assert callable(compiled_fn), "compiler_fn did not return callable"
        self.compiled_fns[id] = compiled_fn


        end = time.perf_counter()
        self.log.debug(f"compile_graph took {(end-start)*1000:.2f}ms")
        return "Finished compiling"

    def call(self, id, *args):
        self.log.info(f"Calling forward on actor {self.id} with {len(args)} args")
        start = time.perf_counter()

        def unwrap(x):
            if isinstance(x, list):
                assert len(x) == 1
            return x[0] if isinstance(x, list) else x
        args = list(map(unwrap, args))

        self.prev_activation = args[0]
        if torch.is_floating_point(self.prev_activation):
            self.prev_activation.requires_grad_()

        # new_args = list(args)
        # all_args = []
        # if self.parameters:
        #     for arg in self.parameters:
        #         if arg is not None:
        #             all_args.append(arg)
        #         else:
        #             all_args.append(new_args[0])
        #             del new_args[0]
        # else:
        #     all_args = new_args

        out = self.compiled_fns[id](*args)

        act = [t for t in out if t.requires_grad]
        if act:
            self.activation = act
        else: 
            self.activation = None

        end = time.perf_counter()  
        self.log.debug(f"forward took {(end-start)*1000:.2f}ms")
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
