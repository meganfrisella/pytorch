from enum import Enum
from typing import NamedTuple
import ray
from ray.dag import InputNode, MultiOutputNode

class Task(NamedTuple):
    device_id: int
    stage_id: int
    mb_idx: int
    is_fwd: bool
    upd: bool

class DAGEdge(NamedTuple):
    from_stage: int
    to_stage: int


def get_backward_targets(stage_id: int, dag_edges: list[DAGEdge]):
    return [edge for edge in dag_edges if edge.to_stage == stage_id]


def validate_schedule(schedule: list[list[Task | None]], dag_edges: list[DAGEdge], num_mbs: int) -> None:
    """
    Validate that the schedule respects well-formedness rules and DAG dependencies.
    
    Args:
        schedule: 2D array with one row per device and one column per time step
        dag_edges: List of DAG edges defining stage dependencies
        num_mbs: Number of microbatches in the schedule
        
    Raises:
        ValueError: If the schedule violates any validation rules
    """
    num_stages, num_steps = len(schedule), len(schedule[0]) if schedule else 0
    
    # Check well-formedness: no duplicates, device_id matches row, and all stages present
    all_tasks = set()
    microbatch_tasks = {}  # mb_idx -> set of (stage_id, is_fwd, upd)
    
    for stage_id in range(num_stages):
        for time_step in range(num_steps):
            task = schedule[stage_id][time_step]
            if task is not None:
                # Check device_id matches row
                if task.device_id != stage_id:
                    raise ValueError(
                        f"Task device_id {task.device_id} does not match row {stage_id} "
                        f"at time step {time_step}"
                    )
                
                # Check for duplicates
                task_key = (task.stage_id, task.mb_idx, task.is_fwd, task.upd)
                if task_key in all_tasks:
                    raise ValueError(
                        f"Duplicate task found: stage_id={task.stage_id}, "
                        f"mb_idx={task.mb_idx}, is_fwd={task.is_fwd}, upd={task.upd}"
                    )
                all_tasks.add(task_key)
                
                # Track tasks by microbatch
                if task.mb_idx not in microbatch_tasks:
                    microbatch_tasks[task.mb_idx] = set()
                microbatch_tasks[task.mb_idx].add((task.stage_id, task.is_fwd, task.upd))
    
    # Get all required stages from DAG edges
    all_required_stages = set()
    for edge in dag_edges:
        all_required_stages.add(edge.from_stage)
        all_required_stages.add(edge.to_stage)
    
    # Check that each microbatch has all required forward and backward stages
    for mb_idx, tasks in microbatch_tasks.items():
        # Find all stages that have forward/backward tasks for this microbatch
        fwd_stages = {stage_id for stage_id, is_fwd, upd in tasks if is_fwd and not upd}
        bwd_stages = {stage_id for stage_id, is_fwd, upd in tasks if not is_fwd and not upd}
        
        # Check that all required stages have forward tasks
        missing_fwd = all_required_stages - fwd_stages
        if missing_fwd:
            raise ValueError(
                f"Microbatch {mb_idx} missing forward stages: {missing_fwd}"
            )
        
        # Check that all required stages have backward tasks
        missing_bwd = all_required_stages - bwd_stages
        if missing_bwd:
            raise ValueError(
                f"Microbatch {mb_idx} missing backward stages: {missing_bwd}"
            )
    
    # Check pipeline stage dependencies
    for mb_idx in range(num_mbs):
        # Find all tasks for this microbatch
        fwd_times = {}  # stage_id -> time_step
        bwd_times = {}  # stage_id -> time_step
        
        for stage_id in range(num_stages):
            for time_step in range(num_steps):
                task = schedule[stage_id][time_step]
                if task is not None and task.mb_idx == mb_idx:
                    if task.is_fwd and not task.upd:
                        fwd_times[task.stage_id] = time_step
                    elif not task.is_fwd and not task.upd:
                        bwd_times[task.stage_id] = time_step
        
        # Check forward stage ordering: if A -> B, then fwd(A) < fwd(B)
        for edge in dag_edges:
            from_stage, to_stage = edge.from_stage, edge.to_stage
            if from_stage in fwd_times and to_stage in fwd_times:
                if fwd_times[from_stage] >= fwd_times[to_stage]:
                    raise ValueError(
                        f"Forward stage ordering violation for microbatch {mb_idx}: "
                        f"forward stage {from_stage} (time {fwd_times[from_stage]}) must come "
                        f"before forward stage {to_stage} (time {fwd_times[to_stage]})"
                    )
        
        # Check forward-backward ordering: fwd(A) < bwd(A)
        for stage_id in fwd_times:
            if stage_id in bwd_times:
                if fwd_times[stage_id] >= bwd_times[stage_id]:
                    raise ValueError(
                        f"Forward-backward ordering violation for microbatch {mb_idx}, "
                        f"stage {stage_id}: forward (time {fwd_times[stage_id]}) must come "
                        f"before backward (time {bwd_times[stage_id]})"
                    )
        
        # Check backward stage ordering: if A -> B, then bwd(B) < bwd(A)
        for edge in dag_edges:
            from_stage, to_stage = edge.from_stage, edge.to_stage
            if from_stage in bwd_times and to_stage in bwd_times:
                if bwd_times[to_stage] >= bwd_times[from_stage]:
                    raise ValueError(
                        f"Backward stage ordering violation for microbatch {mb_idx}: "
                        f"backward stage {to_stage} (time {bwd_times[to_stage]}) must come "
                        f"before backward stage {from_stage} (time {bwd_times[from_stage]})"
                    )

def piper_exec(model, schedule, params, truth, loss_fn, num_mbs):
    assert (
        hasattr(model, "_ray_actors")
        and "model must be compiled with torch.compile with flag distribute=True"
    )
    num_steps, num_stages = len(schedule[0]), len(schedule)
    actors = model._ray_actors
    fwd_fns = model._piper_fwd_fns
    # begin_stage_fns = model._piper_begin_stage

    dag_edges = model._dag
    dag_edges = list(map(lambda e: (DAGEdge(e[0], e[1])), list(model._dag)))
    # print(f"[PIPER] DAG edges: {dag_edges}")
    
    # Validate the schedule before execution
    validate_schedule(schedule, dag_edges, num_mbs)

    # maps mb_idx to the ref resulting from a forward call on the microbatch
    fwd_refs = dict()

    # maps mb_idx to a dict that maps stage_id to the refs output from the stage's backward on that microbatch
    bwd_ref_dicts = dict()

    # iterate over evrery task in the schedule
    ret = []
    for i in range(num_steps):
        for j in range(num_stages-1, -1, -1):
            task = schedule[j][i]
            if task:
                device_id,stage_id, mb_idx, is_fwd, upd = task
                torch._dynamo.eval_frame.dynamo_tls.current_mb = mb_idx
                actor_id = j
                if upd:
                    num_bwd_targets = len(get_backward_targets(stage_id, dag_edges))
                    if num_bwd_targets == 0:
                        num_bwd_targets = 1

                    done_refs = set()
                    for _, bwd_ref_dict in bwd_ref_dicts.items():
                        bwd_refs = bwd_ref_dict[stage_id]
                        bwd_refs = bwd_refs[num_bwd_targets:]
                        if isinstance(bwd_refs, list):
                            done_refs = done_refs | set(bwd_refs)
                        else:
                            done_refs.add(bwd_refs)
                    done_refs = list(done_refs)
                    # print(f"[PIPER] Update stage {stage_id}")
                    upd = actors[actor_id].update.remote(*done_refs)
                    ret.append(upd)
                elif is_fwd:
                    if fwd_fns:
                        # print(f"[PIPER] Forward stage {stage_id} mb {mb_idx}")
                        # Set the current microbatch index in thread-local storage
                        if stage_id == 0:
                            refs = fwd_fns[1](fwd_fns[0](model._orig_mod, *params, dynamo_mb=mb_idx))
                            fwd_refs[mb_idx] = refs
                        else:
                            refs = fwd_refs[mb_idx]
                            if isinstance(refs, list) or isinstance(refs, tuple):
                                refs = fwd_fns[stage_id+1](*refs)
                            else:
                                refs = fwd_fns[stage_id+1](refs)
                            fwd_refs[mb_idx] = refs
                    else:
                        # if this is the first forward task for a microbatch, dispatch the forward task
                        # LIMITATION: forward for all stages is dispatched by this call, cannot interleave
                        # forward tasks with other tasks.
                        if mb_idx not in fwd_refs:
                            # print(f"Fwd mb {mb_idx}")
                            fwd_refs[mb_idx] = model(*params, dynamo_mb=mb_idx)
                else:
                    # log order of task dispatch by printing
                    # also see output_graph.py:1785 where we log forward dispatch
                    # print(f"Calling backward stage {stage_id} mb {mb_idx}")
                    # print(f"[PIPER] Stage {stage_id} backward targets: {get_backward_targets(stage_id, dag_edges)}")
                    num_bwd_targets = len(get_backward_targets(stage_id, dag_edges))
                    if mb_idx not in bwd_ref_dicts:
                        # if this is the first backward task for a microbatch, dispatch the
                        # backward task and cache the resulting ref(s)
                        fwd_ref = fwd_refs[mb_idx]
                        bwd_ref_dicts[mb_idx] = dict()
                        # print(f"[PIPER] Backward stage {stage_id} mb {mb_idx} in: {fwd_ref}")
                        bwd_ref_dicts[mb_idx][stage_id] = (
                            actors[actor_id]
                            .backward.options(num_returns=num_bwd_targets*2)
                            .remote(
                                stage_id, mb_idx, fwd_ref.get_ref(), loss_fn=loss_fn
                            )
                        )
                    else:
                        # if this is not the first backward task for a microbatch, look up
                        # the input ref which represents a gradient from the backward call
                        # of a subsequent stage

                        # get the result of the subsequent stage's backward call
                        # LIMITATION: there cannot be more than one subsequent stage (e.g. the
                        # forward of stage A cannot be inputs to both stage B and C)
                        # this limitation should eventually be resolved by making the logic below more general
                        to_stage = [
                            edge.to_stage
                            for edge in dag_edges
                            if edge.from_stage == stage_id
                        ]
                        assert len(to_stage) == 1
                        to_stage = to_stage[0]

                        # get the refs resulting from the subsequent stage's backward
                        targets = get_backward_targets(to_stage, dag_edges)

                        # get the idx of the current stage in the subsequent stage's output list
                        # LIMITATION: this logic assumes that the user's dag_edges list is ordered according
                        # to how the stages are ordered in the model file. e.g. in clip.py stage 0 comes
                        # before stage 1, so dag_edges must look like
                        # dag_edges = [DAGEdge(0, 2), DAGEdge(1, 2)] and NOT
                        # dag_edges = [DAGEdge(1, 2), DAGEdge(0, 2)]
                        idx = targets.index(DAGEdge(stage_id, to_stage))

                        # make sure the subsequent stage's backward was already dispatched
                        assert to_stage in bwd_ref_dicts[mb_idx]

                        # if the subsequent stage's backward had more than one output, index the
                        # list to get the ref for the current stage
                        bwd_refs = bwd_ref_dicts[mb_idx][to_stage]
                        bwd_ref = bwd_refs[idx]
                        # dispatch the current stage's backward and cache the resulting ref(s)
                        if num_bwd_targets == 0:
                            num_bwd_targets = 1
                        # print(f"[PIPER] Backward stage {stage_id} mb {mb_idx}")
                        bwd_ref_dicts[mb_idx][stage_id] = (
                            actors[actor_id]
                            .backward.options(num_returns=num_bwd_targets*2)
                            .remote(stage_id, mb_idx, bwd_ref)
                        )
    return ret





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
        if isinstance(ins.argval, str) and ins.argval.startswith("__resume_at_"):
            return ins.argval
        # if ins.opname.startswith("CALL"):
        #     # search backward a short window for a LOAD_GLOBAL/LOAD_NAME callee
        #     for back in range(i - 1, max(i - 12, -1), -1):
        #         prev = instrs[back]
        #         if prev.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
        #             if isinstance(prev.argval, str) and prev.argval.startswith("__resume_at_"):
        #                 return prev.argval
        #             break
    return None

def _should_split(code: Bytecode) -> bool:
    """
    Return True iff there is a LOAD_ATTR distributed_stage before
    a call to a __resume_at_* global.
    """
    for instr in code:
        if hasattr(instr, "name") and hasattr(instr, "arg"):
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
        # allow chained form: single CallMarker positional argument
        if len(args) == 1 and not kwargs and isinstance(args[0], CallMarker):
            call_args, call_kwargs = args[0].args, args[0].kwargs
        else:
            call_args, call_kwargs = args, kwargs

        # if no resume_name found, just invoke the function (still accept marker)
        if not should_split:
            return fn(*call_args, **call_kwargs)
        
        # print(f"Stubbing {fn}")

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


import torch
from torch._dynamo.backends.debugging import eager

def piper_setup(model, example_inputs, dynamic=False, backend=None):

    if not backend:
        backend = eager

    # print("[PIPER] Partial output before compiling:")
    # print(model(*example_inputs)[0][0][0])
    
    compiled = torch.compile(model, distribute=True, dynamic=dynamic, backend=backend)

    print("[PIPER] Compiling...")

    out = compiled(*example_inputs)

    print("[PIPER] Compiled...")
    
    # print("[PIPER] Partial output after compiling:")
    # print(out[0][0][0])
    
    import inspect
    # output_codes = torch._dynamo.symbolic_convert.CONT_REGISTRY
    output_codes = torch._dynamo.convert_frame.output_codes
    transformed_fns = {}
    for name, fn in output_codes.piper_fns.items():
        # if name != "forward":
        #     name = output_codes.name_map[name]
        transformed_fns[name] = split_before(fn)

    out = (model, *example_inputs)
    next_call = "forward"
    fwd_fns = []
    while next_call:
        fn = transformed_fns[next_call]
        fwd_fns.append(fn)
        if next_call == "forward":
            out = fn(*out, dynamo_mb=999)
        elif isinstance(out, list) or isinstance(out, tuple):
            out = fn(*out)
        else:
            out = fn(out)
        if isinstance(out, CallMarker):
            next_call = out.next_call
        else:
            next_call = None
    

    # print("[PIPER] Extracted forward functions:")
    # print(fwd_fns)
    # print(torch._dynamo.eval_frame.dynamo_tls.stage_fns)

    setattr(compiled, "_piper_fwd_fns", fwd_fns)
    # setattr(compiled, "_piper_begin_stage", torch._dynamo.eval_frame.dynamo_tls.stage_fns)

    # MEMORY CLEANUP
    torch._dynamo.convert_frame.output_codes.piper_fns.clear()
    torch._dynamo.convert_frame.output_codes.name_map.clear()
    import gc
    gc.collect()

    torch._dynamo.eval_frame.dynamo_tls.currently_compiling = False

    from ray.experimental.collective import create_collective_group
    create_collective_group(
        list(compiled._ray_actors.values()),
        backend="nccl")
    
    print("[PIPER] Started NCCL group")

    return compiled
