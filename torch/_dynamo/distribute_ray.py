import ray
import torch
import logging

import os
import time

@ray.remote
class StageActor:
    # def __init__(self, id, compiler_fn, example_inputs, parameters, optim_fn=None):
    def __init__(self, id, optim_fn=None):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

        self.log.info(f"PID: {os.getpid()}")

        self.log.debug(f"Initializing Ray actor {id}")
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
        self.log.info(f"__init__ took {(end-start)*1000:.2f}ms")

    def id(self):
        return self.id
    g
    def compile_graph(self, id, gm_data, compiler_fn, example_inputs):
        start = time.perf_counter()

        self.gm = torch.load(gm_data, weights_only=False)
        compiled_fn = compiler_fn(self.gm, example_inputs)
        assert callable(compiled_fn), "compiler_fn did not return callable"
        self.compiled_fns[id] = compiled_fn

        end = time.perf_counter()
        self.log.info(f"compile_graph took {(end-start)*1000:.2f}ms")
        return "Finished compiling"

    def forward(self, id, *args):
        self.log.debug(f"Calling forward on actor {self.id} with {len(args)} args")
        start = time.perf_counter()

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
        self.log.info(f"forward took {(end-start)*1000:.2f}ms")
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
