import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from collections import defaultdict


def estimate_model_size(model, example_input):
    # parameters and gradients
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    grad_size = param_size  # one .grad per param
    
    # optimizer states: assume Adam
    optim_size = 2 * param_size
    
    # activations
    acts_size = 0
    handles = []
    def hook(module, inp, out):
        nonlocal acts_size
        if torch.is_tensor(out):
            acts_size += out.nelement() * out.element_size()
        elif isinstance(out, (tuple, list)):
            for t in out:
                if torch.is_tensor(t):
                    acts_size += t.nelement() * t.element_size()
    
    for m in model.modules():
        handles.append(m.register_forward_hook(hook))
    
    # run a dry forward pass
    with torch.no_grad():
        model(example_input)
    
    for h in handles:
        h.remove()
    
    total = param_size + grad_size + optim_size + acts_size
    return {
        "parameters_GB": param_size / 1024**3,
        "gradients_GB": grad_size / 1024**3,
        "optimizer_GB": optim_size / 1024**3,
        "activations_GB": acts_size / 1024**3,
        "total_GB": total / 1024**3,
    }

import torch.fx as fx
# from torch.autograd import profiler as autograd_profiler
from torch.profiler import profile, ProfilerActivity, record_function


def split_fx_graph_at_node(gm: fx.GraphModule, split_node_name: str) -> tuple[fx.GraphModule, fx.GraphModule]:
    """
    Split a torch.fx GraphModule into two separate GraphModules at a specified node.
    
    Args:
        gm: The original GraphModule to split
        split_node_name: The name of the node where to split the graph
        
    Returns:
        Tuple of (first_graph, second_graph) GraphModules
        
    The split node will be the first node in the second graph. Any nodes from the first
    graph that are required by the second graph will be returned as outputs from the
    first graph and passed as inputs to the second graph.
    """
    # Find the split node
    split_node = None
    for node in gm.graph.nodes:
        if node.name == split_node_name:
            split_node = node
            break
    
    if split_node is None:
        raise ValueError(f"Node '{split_node_name}' not found in graph")
    
    # Get all nodes in the first graph (up to split node)
    first_graph_nodes = set()
    for node in gm.graph.nodes:
        if node.name == split_node_name:
            break
        first_graph_nodes.add(node.name)
    
    # Find dependencies of the second graph that are in the first graph
    second_graph_deps = set()
    second_graph = False
    for node in gm.graph.nodes:
        if node.name == split_node_name:
            second_graph = True
        if second_graph:
            for arg in node.all_input_nodes:
                if arg.name in first_graph_nodes:
                    second_graph_deps.add(arg.name)
    
    # Create new graphs
    graph1 = fx.Graph()
    graph2 = fx.Graph()
    
    # Maps old node names to new nodes in each graph
    node_map1 = {}
    node_map2 = {}
    
    # Copy nodes to first graph (up to but not including split node)
    first_graph_outputs = []
    for node in gm.graph.nodes:
        if node.name == split_node_name:
            break
            
        def map_arg(arg):
            if hasattr(arg, 'name'):
                return node_map1[arg.name]
            else:
                return arg
                
        new_args = fx.graph.map_arg(node.args, map_arg)
        new_kwargs = fx.graph.map_arg(node.kwargs, map_arg)
        new_node = graph1.create_node(
            op=node.op,
            target=node.target,
            args=new_args,
            kwargs=new_kwargs,
            name=node.name
            )
        node_map1[node.name] = new_node
        
        # If this node is needed by the second graph, add it to outputs
        if node.name in second_graph_deps:
            first_graph_outputs.append(node_map1[node.name])
    
    # Set outputs for first graph
    if first_graph_outputs:
        graph1.output(first_graph_outputs[0] if len(first_graph_outputs) == 1 else first_graph_outputs)
    
    # Create placeholders in second graph for dependencies from first graph
    dep_placeholders = {}
    for node in first_graph_outputs:
        name = node.name
        placeholder = graph2.create_node(
            op="placeholder",
            target=name,
            name=f"{name}_input"
        )
        dep_placeholders[name] = placeholder
        node_map2[f"{name}_input"] = placeholder
    
    # Copy nodes to second graph (starting from split node)
    copy_to_second = False
    for node in gm.graph.nodes:
        if node.name == split_node_name:
            copy_to_second = True
            
        if not copy_to_second:
            continue
            
        if node.op == "placeholder":
            # Skip original placeholders in second graph
            raise RuntimeError(f"Placeholder node found in second graph: {node.name}")

        # Map arguments: use new nodes if available, otherwise use dependency placeholders
        def map_arg(arg):
            if hasattr(arg, 'name') and arg.name in node_map2:
                return node_map2[arg.name]
            elif hasattr(arg, 'name'):
                return dep_placeholders[arg.name]
            else:
                return arg
        
        new_args = fx.graph.map_arg(node.args, map_arg)
        new_kwargs = fx.graph.map_arg(node.kwargs, map_arg)
        new_node = graph2.create_node(
            op=node.op,
            target=node.target,
            args=new_args,
            kwargs=new_kwargs,
            name=node.name
        )
        node_map2[node.name] = new_node
    
    # Create new GraphModules
    gm1 = fx.GraphModule(gm, graph1)
    gm2 = fx.GraphModule(gm, graph2)
    
    return gm1, gm2


def split_fx_graph_at_nodes(gm: fx.GraphModule, split_node_names: list[str]) -> list[fx.GraphModule]:
    """
    Split a GraphModule into N sequential GraphModules given N-1 split node names.

    The first returned module contains nodes before the first split, the last
    contains nodes from the last split to the end. Inter-stage dependencies are
    returned as outputs from the previous stage and become placeholders in the next.
    """
    if not split_node_names:
        return [gm]

    modules: list[fx.GraphModule] = []
    current_gm = gm
    for split_name in split_node_names:
        left, right = split_fx_graph_at_node(current_gm, split_name)
        modules.append(left)
        current_gm = right
    modules.append(current_gm)
    return modules


def _resolve_attr_path(root, path: str):
    """Resolve a dotted attribute path on a module/GraphModule.

    Supports numeric indices (e.g., 'layers.0.attention_norm.weight').
    """
    # First, try direct parameter/buffer resolution on GraphModule
    try:
        return root.get_parameter(path)
    except Exception:
        pass
    try:
        return root.get_buffer(path)
    except Exception:
        pass

    # Fallback: traverse attributes, supporting numeric indices on sequence-like modules
    obj = root
    for part in str(path).split('.'):
        if hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        if part.isdigit():
            if isinstance(obj, (nn.ModuleList, nn.Sequential, list, tuple)):
                obj = obj[int(part)]
                continue
        raise AttributeError(f"Cannot resolve path segment '{part}' in '{path}' from object type {type(obj)}")
    return obj

def profile_fx_fw_bw_old(model: nn.Module, example_input, loss_fn=None, device="cuda"):
    model = model.to(device)
    gm = fx.symbolic_trace(model)

    gm1, gm2 = split_fx_graph_at_node(gm, "float_11") # float_71

    if loss_fn is None:
        loss_fn = lambda y: y.sum()

    def profile_gm(graph_module: fx.GraphModule, stage_name: str, input_data, fake_grad=None, grad_index=None):
        """Profile a single GraphModule and return stats and profiler."""
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        ) as prof:
            # Profile forward pass
            with record_function(f"FX_{stage_name}_Forward"):
                if isinstance(input_data, (tuple, list)):
                    outputs = graph_module(*input_data)
                else:
                    outputs = graph_module(input_data)

            # Profile backward pass
            with record_function(f"FX_{stage_name}_Backward"):
                if fake_grad is not None:
                    # For stage 1: use fake gradient from stage 2 on the specific output
                    assert isinstance(outputs, (tuple, list))
                    outputs[grad_index].backward(fake_grad, retain_graph=True)
                else:
                    # For stage 2: compute loss and backward
                    loss = loss_fn(outputs)
                    loss.backward(retain_graph=True)

        # Extract profiler stats - aggregate across multiple entries per function
        stats = {}
        for evt in prof.key_averages():  # no shape grouping
            if not evt.key.startswith("FX_"):
                continue
            name = evt.key.replace("FX_", "")

            # Read metrics
            cpu_total = getattr(evt, "cpu_time_total", 0.0)
            cuda_total = getattr(evt, "device_time_total", 0.0)
            cpu_mem = getattr(evt, "cpu_memory_usage", 0)
            cuda_mem = getattr(evt, "device_memory_usage", 0)

            # Initialize or aggregate stats for this function
            if name not in stats:
                stats[name] = {
                    "cpu_total_ms": 0.0,
                    "cuda_total_ms": 0.0,
                    "cpu_mem_MB": 0.0,
                    "cuda_mem_MB": 0.0,
                }
            
            # Aggregate metrics (take max across all entries)
            stats[name]["cpu_total_ms"] = max(stats[name]["cpu_total_ms"], cpu_total / 1e3)
            stats[name]["cuda_total_ms"] = max(stats[name]["cuda_total_ms"], cuda_total / 1e3)
            stats[name]["cpu_mem_MB"] = max(stats[name]["cpu_mem_MB"], cpu_mem / 1024**2)
            stats[name]["cuda_mem_MB"] = max(stats[name]["cuda_mem_MB"], cuda_mem / 1024**2)

        return stats, prof
    
    x = example_input.to(device)


    # Profile original graph
    for _ in range(10):
        # Warmup original graph
        orig_outputs = gm(x)
        loss = loss_fn(orig_outputs)
        loss.backward()

    original_stats, original_prof = profile_gm(gm, "Original", x)
    

    # Profile first split graph - need to create fake gradient for backward pass
    for _ in range(10):
        # Warmup stage1 graph
        stage1_outputs = gm1(x)
        grad_required_indices = [i for i, out in enumerate(stage1_outputs) if out.requires_grad]
        if len(grad_required_indices) == 1:
            grad_index = grad_required_indices[0]
            grad_required_output = stage1_outputs[grad_index]
            fake_grad = torch.ones_like(grad_required_output)
            stage1_outputs[grad_index].backward(fake_grad)

    stage1_outputs = gm1(x)
    # Check which outputs require gradients and track their indices
    grad_required_indices = [i for i, out in enumerate(stage1_outputs) if out.requires_grad]
    
    if len(grad_required_indices) == 0:
        raise RuntimeError("No outputs from stage 1 require gradients")
    elif len(grad_required_indices) > 1:
        raise RuntimeError(f"Multiple outputs from stage 1 require gradients: {len(grad_required_indices)}")
    
    # Get the index and output that requires gradients
    grad_index = grad_required_indices[0]
    grad_required_output = stage1_outputs[grad_index]
    
    # Create fake gradient with same shape as the output that requires grad
    fake_grad = torch.ones_like(grad_required_output)
    
    stage1_stats, stage1_prof = profile_gm(gm1, "Stage1", x, fake_grad, grad_index)
    

    # Profile second split graph
    for _ in range(10):
        # Warmup stage2 graph
        stage2_outputs = gm2(*stage1_outputs)
        loss = loss_fn(stage2_outputs)
        loss.backward()
    stage2_stats, stage2_prof = profile_gm(gm2, "Stage2", stage1_outputs)

    # Combine all stats
    all_stats = {
        "original": original_stats,
        "stage1": stage1_stats,
        "stage2": stage2_stats
    }

    return all_stats, (original_prof, stage1_prof, stage2_prof)


def profile_fx_fw_bw(model: nn.Module, example_input, split_nodes: list[str], loss_fn=None, device="cuda", num_iters=10):
    """Profile FX graphs using CUDA events and torch.cuda.memory_allocated for timing and memory.

    Args:
        model: PyTorch module to profile.
        example_input: Sample input tensor(s).
        loss_fn: Loss function for final stage; defaults to sum.
        device: Device to place the model and inputs on.
        num_iters: Number of averaged profiling iterations per stage.
        split_nodes: Optional list of node names to split at (length = N-1) to produce N stages.
    """
    model = model.to(device)
    gm = fx.symbolic_trace(model)

    stage_modules = split_fx_graph_at_nodes(gm, split_nodes)

    if loss_fn is None:
        loss_fn = lambda y: y.sum()

    def profile_gm(graph_module: fx.GraphModule, stage_name: str, input_data, fake_grad=None, grad_index=None):
        """Profile a single GraphModule using CUDA events and memory tracking over multiple iterations."""

        # set up an optimizer to time the optimizer step
        optimizer = torch.optim.Adam(graph_module.parameters())
        
        # Create CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warmup run
        with torch.no_grad():
            if isinstance(input_data, (tuple, list)):
                _ = graph_module(*input_data)
            else:
                _ = graph_module(input_data)
        
        torch.cuda.synchronize()
        
        # Accumulate statistics over multiple iterations
        forward_times = []
        forward_peak_memories = []
        forward_peak_memory_deltas = []
        backward_times = []
        backward_peak_memories = []
        backward_peak_memory_deltas = []
        update_times = []
        update_peak_memories = []
        update_peak_memory_deltas = []
        for _ in range(num_iters):
            # Record initial memory state for this iteration
            torch.cuda.synchronize()
            
            # Reset peak memory counter for forward pass
            torch.cuda.reset_peak_memory_stats()
            forward_start_memory = torch.cuda.memory_allocated()

            # Profile forward pass
            start_event.record()
            
            if isinstance(input_data, (tuple, list)):
                outputs = graph_module(*input_data)
            else:
                outputs = graph_module(input_data)
            
            end_event.record()
            torch.cuda.synchronize()
            
            forward_time_ms = start_event.elapsed_time(end_event)
            forward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - forward_start_memory) / (1024**3)
            forward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            
            forward_times.append(forward_time_ms)
            forward_peak_memories.append(forward_peak_memory_gb)
            forward_peak_memory_deltas.append(forward_peak_memory_delta_gb)
            
            # Reset peak memory counter for backward pass
            torch.cuda.reset_peak_memory_stats()
            backward_start_memory = torch.cuda.memory_allocated()
            
            # Profile backward pass
            start_event.record()
            
            if fake_grad is not None:
                # For intermediate stage N (not final stage): use fake gradient from stage N+1
                if isinstance(outputs, (tuple, list)):
                    assert grad_index is not None
                    outputs[grad_index].backward(fake_grad, retain_graph=True)
                else:
                    assert outputs.requires_grad
                    outputs.backward(fake_grad, retain_graph=True)
            else:
                # For final stage: compute loss and backward
                loss = loss_fn(outputs)
                loss.backward(retain_graph=True)
            
            end_event.record()
            torch.cuda.synchronize()
            
            backward_time_ms = start_event.elapsed_time(end_event)
            backward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - backward_start_memory) / (1024**3)
            backward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            
            backward_times.append(backward_time_ms)
            backward_peak_memories.append(backward_peak_memory_gb)
            backward_peak_memory_deltas.append(backward_peak_memory_delta_gb)

            # Reset peak memory counter for optimizer step
            torch.cuda.reset_peak_memory_stats()
            update_start_memory = torch.cuda.memory_allocated()
            
            # Profile optimizer step
            start_event.record()
            optimizer.step()
            end_event.record()
            torch.cuda.synchronize()
            
            update_time_ms = start_event.elapsed_time(end_event)
            update_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - update_start_memory) / (1024**3)
            update_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            
            update_times.append(update_time_ms)
            update_peak_memories.append(update_peak_memory_gb)
            update_peak_memory_deltas.append(update_peak_memory_delta_gb)
        
        # Remove the first 10 entries from each list to account for warmup
        WARMUP_COUNT = 10
        forward_times = forward_times[WARMUP_COUNT:]
        forward_peak_memories = forward_peak_memories[WARMUP_COUNT:]
        forward_peak_memory_deltas = forward_peak_memory_deltas[WARMUP_COUNT:]
        backward_times = backward_times[WARMUP_COUNT:]
        backward_peak_memories = backward_peak_memories[WARMUP_COUNT:]
        backward_peak_memory_deltas = backward_peak_memory_deltas[WARMUP_COUNT:]
        update_times = update_times[WARMUP_COUNT:]
        update_peak_memories = update_peak_memories[WARMUP_COUNT:]
        update_peak_memory_deltas = update_peak_memory_deltas[WARMUP_COUNT:]

        # Calculate average statistics
        avg_forward_time = sum(forward_times) / len(forward_times)
        avg_forward_peak_memory = sum(forward_peak_memories) / len(forward_peak_memories)
        avg_forward_peak_memory_delta = sum(forward_peak_memory_deltas) / len(forward_peak_memory_deltas)
        avg_backward_time = sum(backward_times) / len(backward_times)
        avg_backward_peak_memory = sum(backward_peak_memories) / len(backward_peak_memories)
        avg_backward_peak_memory_delta = sum(backward_peak_memory_deltas) / len(backward_peak_memory_deltas)
        avg_update_time = sum(update_times) / len(update_times)
        avg_update_peak_memory = sum(update_peak_memories) / len(update_peak_memories)
        avg_update_peak_memory_delta = sum(update_peak_memory_deltas) / len(update_peak_memory_deltas)

        stats = {
            f"{stage_name}_Forward": {
                "cuda_time_ms": avg_forward_time,
                "cuda_mem_peak_GB": avg_forward_peak_memory,
                "cuda_mem_delta_GB": avg_forward_peak_memory_delta,
            },
            f"{stage_name}_Backward": {
                "cuda_time_ms": avg_backward_time,
                "cuda_mem_peak_GB": avg_backward_peak_memory,
                "cuda_mem_delta_GB": avg_backward_peak_memory_delta,
            },
            f"{stage_name}_Update": {
                "cuda_time_ms": avg_update_time,
                "cuda_mem_peak_GB": avg_update_peak_memory,
                "cuda_mem_delta_GB": avg_update_peak_memory_delta,
            },
        }
        
        return stats
    
    x = example_input.to(device)

    # Profile original graph
    original_stats = profile_gm(gm, "Original", x)

    all_stats: dict[str, dict] = {"original": original_stats}

    # Sequentially execute and profile each stage; for intermediate stages feed x, for others feed outputs
    inputs_for_next = x
    for stage_idx, stage_gm in enumerate(stage_modules):
        stage_name = f"Stage{stage_idx}"

        # Run once to discover which outputs require grad (for non-final stages)
        outputs = stage_gm(inputs_for_next) if not isinstance(inputs_for_next, (tuple, list)) else stage_gm(*inputs_for_next)

        if stage_idx < len(stage_modules) - 1:
            if isinstance(outputs, (tuple, list)):
                grad_required_indices = [i for i, out in enumerate(outputs) if torch.is_tensor(out) and out.requires_grad]
                if len(grad_required_indices) == 0:
                    raise RuntimeError(f"No outputs from stage {stage_idx} require gradients")
                if len(grad_required_indices) > 1:
                    raise RuntimeError(f"Multiple outputs from stage {stage_idx} require gradients: {len(grad_required_indices)}")
                grad_index = grad_required_indices[0]
                fake_grad = torch.ones_like(outputs[grad_index])
                stage_stats = profile_gm(stage_gm, stage_name, inputs_for_next, fake_grad, grad_index)
            else:
                assert outputs.requires_grad
                fake_grad = torch.ones_like(outputs)
                stage_stats = profile_gm(stage_gm, stage_name, inputs_for_next, fake_grad)
        else:
            # Final stage: compute loss over its outputs
            stage_stats = profile_gm(stage_gm, stage_name, inputs_for_next)

        all_stats[f"stage{stage_idx}"] = stage_stats
        # detach to simulate transfering raw tensors between devices
        inputs_for_next = [out.detach() for out in outputs] if isinstance(outputs, (tuple, list)) else outputs.detach()

    return all_stats