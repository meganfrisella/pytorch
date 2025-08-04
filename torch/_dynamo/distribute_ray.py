import ray
import torch
import logging

import os
import time
import uuid

import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch._guards import CompileId

_fake_tensor_mode = FakeTensorMode()
_fake_tensor_converter = _fake_tensor_mode.fake_tensor_converter

torch.set_float32_matmul_precision('high')

class RemoteTensorKey:
    def __init__(self):
        self.key = str(uuid.uuid4())


def get_fake_tensors(example_inputs, graph_module: fx.GraphModule):
    fake_inputs = []
    for inp in example_inputs:
        if isinstance(inp, torch.Tensor):
            fake_inputs.append(_fake_tensor_converter.from_real_tensor(_fake_tensor_mode, inp))
        else:
            fake_inputs.append(inp)
    with _fake_tensor_mode:
        fakes = graph_module.forward(*fake_inputs)
    return fakes


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
            if isinstance(obj, list):
                assert len(obj) == 1
                self.resolved = obj[0]
            else:
                self.resolved = obj
        return self.resolved
    
    def get_ref(self):
        return self.obj_ref

    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(x):
            return x.get() if isinstance(x, RemoteTensor) else x
        args = list(map(unwrap, args))
        kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        out = func(*args, **kwargs)
        return out
    
    def __repr__(self):
        return f"RemoteTensor(obj_ref={self.obj_ref})"


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

        self.input = None
        self.truth = None

        # ordered list of frame ids for ordering the fx.Graphs on this actor
        self.frame_ids = []
        # map compile id -> compiled fx.Graph function
        self.compiled_fns = dict()
        # map compile id -> model parameters used by the fx.Graph
        self.parameters = dict()
        # map compile id -> optimizer for the fx.Graph
        self.optims = dict()
        # map mb_idx -> previous activation (if this stage is not first)
        self.prev_activations = dict()
        # map mb_idx -> current activation
        self.activations = dict()

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

    def compile_graph(self, compile_id: CompileId, gm_data, compiler_fn, example_inputs, parameters):
        self.log.info(f"Compiling graph on actor {self.actor_id}. compile id: {compile_id}. inputs: {len(example_inputs)}")
        start = time.perf_counter()

        # if this is a recompile, assert the non-null parameters have the same shape
        # TODO: this is a weak check
        frame_id = compile_id.frame_id
        if frame_id in self.parameters:
            old_params = self.parameters[frame_id]
            non_null_old_params = [p for p in old_params if p is not None]
            non_null_new_params = [p for p in parameters if p is not None]

            assert len(non_null_old_params) == len(non_null_new_params)
            for p1, p2 in zip(non_null_old_params, non_null_new_params):
                assert p1.shape == p2.shape
            self.log.info(f"Recompiling frame_id {frame_id} on actor {self.actor_id}")

            # save the new null pattern with the original parameters values
            if len(old_params) != len(parameters):
                old_idx = 0
                for new_idx, p in enumerate(parameters):
                    if p is not None:
                        parameters[new_idx] = non_null_old_params[old_idx]
                        old_idx += 1
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
                self.optims[frame_id] = self.optim_fn(non_null_params)
        
        # save the frame id in an ordered list
        if frame_id not in self.frame_ids:
            self.frame_ids.append(frame_id)

        gm = torch.load(gm_data, weights_only=False)
        compiled_fn = compiler_fn(gm, example_inputs)
        assert callable(compiled_fn), "compiler_fn did not return callable"
        self.compiled_fns[compile_id] = compiled_fn

        end = time.perf_counter()
        self.log.debug(f"compile_graph took {(end-start)*1000:.2f}ms")
        return "Finished compiling"

    @ray.method(tensor_transport="nccl")
    def call(self, compile_id: CompileId, mb_idx: int, *args):
        self.log.debug(f"Calling forward {compile_id} mb {mb_idx} on actor {self.actor_id} with {len(args)} args")
        # Ray object refs resolve to a single element list
        def unwrap(x):
            if isinstance(x, list) or isinstance(x, tuple):
                assert len(x) == 1
            return x[0] if isinstance(x, list) or isinstance(x, tuple) else x
        args = list(map(unwrap, args))

        def send_to_device(param):
            if isinstance(param, torch.Tensor):
                return param.to('cuda')
            else:
                return param
        args = list(map(send_to_device, args))

        def pre_loaded_input(param):
            if param is None:
                return self.input
            else:
                return param
        args = list(map(pre_loaded_input, args))

        # save all inputs with gradients as previous activations for backprop
        frame_id = compile_id.frame_id
        if frame_id == self.frame_ids[0]:
            activation_idxs = self.prev_activation_idxs
            prev_activations = []
            for idx, arg in enumerate(args):
                if idx in activation_idxs:
                    if not isinstance(arg, torch.Tensor):
                        print(arg)
                    prev_activations.append(arg.requires_grad_())
            self.prev_activations[mb_idx] = prev_activations

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

        # save all outputs which require gradients as the activations
        if frame_id == self.frame_ids[-1]:
            self.activations[mb_idx] = [t for t in out if t.requires_grad]

        return out

    def call_cpu(self, compile_id: CompileId, mb_idx: int, *args):
        self.log.debug(f"Calling cpu forward {compile_id} mb {mb_idx} on actor {self.actor_id} with {len(args)} args")

        def send_to_device(param):
            if isinstance(param, torch.Tensor):
                return param.to('cuda')
            else:
                return param
        args = list(map(send_to_device, args))

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

        # save all inputs with gradients as previous activations for backprop
        frame_id = compile_id.frame_id
        if frame_id == self.frame_ids[0]:
            activation_idxs = []
            prev_activations = []
            for idx, arg in enumerate(args):
                if isinstance(arg, torch.Tensor) and arg.requires_grad:
                    prev_activations.append(arg)
                    activation_idxs.append(idx)
            self.prev_activations[mb_idx] = prev_activations
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

        # save all outputs which require gradients as the activations
        if frame_id == self.frame_ids[-1]:
            self.activations[mb_idx] = [t for t in out if t.requires_grad]

        return out

    @ray.method(tensor_transport="nccl")
    def backward(self, mb_idx: int, inp, loss_fn=None):
        self.log.debug(f"Calling backward mb {mb_idx} on actor {self.actor_id}")
        activation = self.activations[mb_idx]
        assert activation is not None
        activation = activation[0]
        # compute loss in the last stage. use the saved activation rather
        # than inp because the saved activation remembers the computation graph
        if loss_fn is not None:
            assert self.truth is not None
            loss = loss_fn(activation, self.truth)
            loss.backward()
        # if not the last stage, backprop on the stored activation given 
        # the input gradient from the subsequent stage
        else:
            assert inp is not None
            assert activation.shape == inp.shape
            activation.backward(gradient=inp)

        prev_activations = self.prev_activations[mb_idx]
        if prev_activations:
            ret = [act.grad for act in prev_activations]
        else:
            ret = ["done"]

        return ret + ret

    def update(self, *done_mbs):
        self.log.debug(f"Calling update on actor {self.actor_id}")
        assert self.optim_fn
        for _, optim in self.optims.items():
            optim.step()
            optim.zero_grad()
        return "done"
