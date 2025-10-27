import ray
import torch
import logging
import os
import time
from torch._guards import CompileId
from collections import defaultdict

torch.set_float32_matmul_precision('high')

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

        # Save the parameters and initialize the optimizer
        # TODO: in the case of a recompile, move as few parameters as possible
        frame_id = compile_id.frame_id
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

        gm = torch.piper.utils.deserialize_graphmodule(gm_data)
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
        # for idx, t in enumerate(out):
        #     print(f"STAGE {stage_id} MB {mb_idx} ACT {idx}", t.requires_grad)
        self.activations[stage_id][mb_idx] = [t for t in out if isinstance(t, torch.Tensor) and t.requires_grad]

        # End timing and record data
        if self.tracing:
            print("HERE")
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
        if stage_id != 0:
            activation_idxs = []
            prev_activations = []
            for idx, arg in enumerate(args):
                if isinstance(arg, torch.Tensor) and arg.requires_grad:
                    prev_activations.append(arg)
                    activation_idxs.append(idx)
            self.prev_activations[stage_id][mb_idx] = prev_activations
            self.prev_activation_idxs = activation_idxs

        # patch the args into the stored parameters list, which has None values for the args
        frame_id = compile_id.frame_id
        parameters = self.parameters[frame_id]
        args_plus_parameters = []

        assert len(args) == len([p for p in parameters if p is None])
        if frame_id in self.parameters:
            for arg in self.parameters[frame_id]:
                if arg is not None:
                    args_plus_parameters.append(arg)
                else:
                    args_plus_parameters.append(args[0])
                    del args[0]
        else:
            args_plus_parameters = args


        out = self.compiled_fns[compile_id](*args_plus_parameters)

        del args_plus_parameters
        del args
        del parameters

        # save all outputs which require gradients as the activations
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