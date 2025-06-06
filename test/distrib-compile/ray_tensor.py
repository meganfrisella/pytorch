import ray
import torch
import uuid

import torch.nn as nn
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
import torch.overrides
from torch.utils import _pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode
import logging


class TwoLayerNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
    
    def forward(self, x):
        return self.fc2(self.fc1(x))


_fake_tensor_mode = FakeTensorMode()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


TORCH_TENSOR_CONSTRUCTORS = [
    torch.ops.aten.empty.memory_format,
    torch.ops.aten.randn.default,
]

RETURNS_TENSOR = set(
    [
        torch.ops.aten.sum.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mm.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.t.default,
        torch.ops.aten.relu.default,
        torch.ops.aten.addmm.default,
    ]
)


# Many torch functions are not serializable. Helper to convert a function to a
# name that can be looked up on the actor.
def get_func_name(func):
    if "source_Storage_storage_offset" in func.__name__:
        import ipdb

        ipdb.set_trace()
    if torch.overrides.is_tensor_method_or_property(func):
        return func

    if hasattr(func, "__qualname__"):
        return func.__qualname__
    elif hasattr(func, "__name__"):
        return func.__name__
    else:
        return str(func)


def get_func(func_or_func_name):
    if isinstance(func_or_func_name, str):
        try:
            return _get_func(torch, func_or_func_name)
        except AttributeError:
            return _get_func(torch._C, func_or_func_name)
    else:
        return func_or_func_name


def _get_func(func, func_name):
    import torch.ops

    if len(func_name.split("::")) > 1:
        lib, func_name = func_name.split("::")
        func = getattr(torch.ops, lib)

    func_name = func_name.split(".")
    for attr in func_name:
        func = getattr(func, attr)
    return func


class RemoteTensorKey:
    def __init__(self):
        self.key = str(uuid.uuid4())


class RemoteTensor(torch.Tensor):
    _fake: torch.Tensor

    def __new__(cls, fake: torch.Tensor, actor_handle, func_name, args, kwargs):
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
        instance.actor_handle = actor_handle
        instance._fake = fake
        instance.key = RemoteTensorKey()
        instance.obj_ref = actor_handle.create_tensor.remote(
            instance.key, func_name, args, kwargs
        )
        ray.get(instance.obj_ref)
        return instance

    def get(self):
        return ray.get(self.obj_ref)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        actor_handle = None
        flattened_args, args_spec = pytree.tree_flatten((args, kwargs or {}))
        fake_args = flattened_args[:]
        for i, arg in enumerate(flattened_args):
            if isinstance(arg, cls):
                assert actor_handle is None or actor_handle == arg.actor_handle
                actor_handle = arg.actor_handle
                flattened_args[i] = arg.key
                fake_args[i] = arg._fake
        args, kwargs = pytree.tree_unflatten(flattened_args, args_spec)
        fake_args, fake_kwargs = pytree.tree_unflatten(fake_args, args_spec)

        func_name = get_func_name(func)
        s = f"RemoteTensor.__torch_dispatch__: {func_name} {func}\n\t- args: {tuple(args)}, \n\t- kwargs: {kwargs}\n"
        logger.debug(s)

        # Handle tensor operations that need to maintain autograd metadata
        if func in RETURNS_TENSOR:
            with _fake_tensor_mode:
                fake = func(*fake_args, **fake_kwargs)
            # Create a fresh tensor wrapper to avoid view issues
            return RemoteTensor(fake.detach(), actor_handle, func_name, args, kwargs)

        return ray.get(actor_handle.execute.remote(func_name, args, kwargs))

    def __repr__(self):
        return "ray_" + ray.get(
            self.actor_handle.execute.remote("Tensor.__repr__", (self.key,), {})
        )


@ray.remote
class TensorCreator:
    def __init__(self):
        self.tensors = {}

    def create_tensor(self, key: RemoteTensorKey, func_name, args, kwargs):
        logger.debug("CREATE", key.key)
        tensor = self.execute(func_name, args, kwargs)
        self.tensors[key.key] = tensor
        return self.tensors[key.key]

    def execute(self, func_name, args, kwargs):
        logger.debug(
            f"EXECUTE {func_name}\n\t- args: {tuple(args)}\n\t- kwargs: {kwargs}"
        )
        import torch.ops

        func = get_func(func_name)

        flattened_args, args_spec = pytree.tree_flatten((args, kwargs))
        for i, arg in enumerate(flattened_args):
            if isinstance(arg, RemoteTensorKey):
                flattened_args[i] = self.tensors[arg.key]
            else:
                flattened_args[i] = arg
        args, kwargs = pytree.tree_unflatten(flattened_args, args_spec)
        return_val = func(*args, **kwargs)
        logger.debug(f"\n\t- return val: {return_val}")
        return return_val


class MyDispatchMode(TorchDispatchMode):
    def __init__(self, actor):
        self.actor = actor

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                processed_args.append(f"torch.Tensor[{arg.shape}]")
            else:
                processed_args.append(arg)
        processed_kwargs = {}
        for kw, arg in kwargs.items():
            if isinstance(arg, torch.Tensor):
                processed_kwargs[kw] = f"torch.Tensor[{arg.shape}]"
            else:
                processed_kwargs[kw] = arg
        logger.debug(
            "__torch_dispatch__: func=%s, args=%s, kwargs=%s",
            func,
            tuple(processed_args),
            processed_kwargs,
        )

        if func in TORCH_TENSOR_CONSTRUCTORS:
            # torch.Tensors get created for small values like torch shape.
            # Only offload tensors that have the specified device to the
            # remote worker.
            func_name = get_func_name(func)
            with _fake_tensor_mode:
                fake = func(*args, **kwargs)
            return RemoteTensor(fake, self.actor, func_name, args, kwargs)

        return func(*args, **(kwargs or {}))


def main():
    actor = TensorCreator.remote()

    with MyDispatchMode(actor):
        t = torch.randn((2, 2))
        # print("tensor:", t)
        # print("sum:", torch.sum(t))
        # print("add:", torch.add(t, t))
        # print("prod:", torch.matmul(t, t))

        model = TwoLayerNN(2, 2, 2)
        print("model:", model)
        for p in model.parameters():
            print("param:", p)
        out = model(t)
        print(out)
        out.backward()


if __name__ == "__main__":
    main()
