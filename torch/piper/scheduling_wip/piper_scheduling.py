from typing import NamedTuple, Dict, List, Tuple
from collections import defaultdict


class DAGTask(NamedTuple):
    fwd: bool
    upd: bool
    device: int
    stage: int
    mbtch: int
    duration: int
    mem: float

class DAGEdge(NamedTuple):
    weight: float
    comm: bool

class Task(NamedTuple):
    device_id: int
    stage_id: int
    mb_idx: int
    is_fwd: bool
    upd: bool

class Edge(NamedTuple):
    from_stage: int
    to_stage: int

class DAG:
    def __init__(self, grid_schedule: List[List[Task]], nodes: Dict[str, DAGTask], comm_cost: int):
        # adjacency: key -> {neighbor_key: DAGEdge}
        self.grid_schedule: List[List[Task]] = grid_schedule
        self.nodes: Dict[str, DAGTask] = dict(nodes)
        self._adj: Dict[str, Dict[str, DAGEdge]] = {k: {} for k in nodes}
        self.comm_cost = comm_cost

    def add_edge(self, u: str, v: str):
        if u not in self.nodes or v not in self.nodes:
            raise ValueError("nodes must be in DAG")
        origin = self.nodes[u]
        destination = self.nodes[v]
        comm = origin.device != destination.device
        comm_cost = self.comm_cost if comm else 0
        self._adj[u][v] = DAGEdge(weight=origin.duration + comm_cost, comm=comm)

    def longest_path(self, src: str, dst: str) -> Tuple[List[str], float]:
        """Return the longest path from src to dst as a tuple of (path, value)."""
        # topological sort via Kahn's algorithm
        indeg = {u: 0 for u in self.nodes}
        for u in self._adj:
            for v in self._adj[u]:
                indeg[v] += 1

        topo = []
        q = [u for u, d in indeg.items() if d == 0]
        while q:
            u = q.pop()
            topo.append(u)
            for v in self._adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        
        assert len(topo) == len(self.nodes) # no cycles

        # DP: distance init -inf except src = 0
        dist = {u: float("-inf") for u in self.nodes}
        dist[src] = 0.0
        # Track predecessors to reconstruct path
        pred = {u: None for u in self.nodes}
        
        for u in topo:
            for v, e in self._adj[u].items():
                cand = dist[u] + e.weight
                if cand > dist[v]:
                    dist[v] = cand
                    pred[v] = u

        # Reconstruct path from dst to src
        path = []
        current = dst
        while current is not None:
            path.append(current)
            current = pred[current]
        path.reverse()

        return path, dist[dst] + self.nodes[dst].duration
    
    def activation_memory(self, model_size_stats) -> float:
        total_act_mem = defaultdict(float)
        max_act_mem = defaultdict(float)
        in_flight_mbs = defaultdict(set)
        for device_id, tasks in enumerate(self.grid_schedule):
            for task in tasks:
                if task is not None:
                    if task.is_fwd:
                        in_flight_mbs[device_id].add((task.stage_id, task.mb_idx))
                        total_act_mem[device_id] += model_size_stats["activations_GB"]
                        if total_act_mem[device_id] > max_act_mem[device_id]:
                            max_act_mem[device_id] = total_act_mem[device_id]
                    elif not task.upd:
                        in_flight_mbs[device_id].remove((task.stage_id, task.mb_idx))
                        total_act_mem[device_id] -= model_size_stats["activations_GB"]
        return max_act_mem

def print_schedule(schedule):
    for stage in schedule:
        for task in stage:
            if task:
                string = f"{'f' if task.is_fwd else 'b' if not task.upd else 'u'}_{task.device_id}_{task.stage_id}_{task.mb_idx}"
            else:
                string = "  ---- "
            print(string, end="  ")
        print()

def build_1f1b_schedule(n_mubatches: int, num_stages: int):
    assert num_stages in [2, 4] and "LLAMA must have 2 or 4 stages"
    steps = n_mubatches + num_stages - 1
    schedule = [[None] * (steps * 2 + 1) for _ in range(num_stages)]
    stage_mubatch = [[0, 0] for _ in range(num_stages)]
    for step in range(num_stages):
        for stage_id in range(num_stages):
            if step >= stage_id:
                mubatch_idx = stage_mubatch[stage_id][0]
                if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                    schedule[stage_id][step] = Task(
                        stage_id,stage_id, mubatch_idx, True, False
                    )
                    stage_mubatch[stage_id][0] += 1
    for step in range(num_stages, 2 * steps):
        relative_step = step - num_stages
        for stage_id in range(num_stages):
            inv_stage = num_stages - stage_id - 1
            if relative_step >= inv_stage:
                fwd_or_bwd = 1 - (relative_step + inv_stage) % 2
                task_type = True if fwd_or_bwd == 0 else False
                mubatch_idx = stage_mubatch[stage_id][fwd_or_bwd]
                if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                    schedule[stage_id][step] = Task(
                        stage_id, stage_id, mubatch_idx, task_type, False
                    )
                    stage_mubatch[stage_id][fwd_or_bwd] += 1
    
    for i, stage in enumerate(range(num_stages)):
        schedule[stage][-i-1] = Task(stage_id=stage, device_id=stage, mb_idx=0, is_fwd=False, upd=True)
    # add update tasks and make schedule optimizations
    # if num_stages == 4:
    #     schedule[0][-1] = Task(device_id=0, stage_id=0, mb_idx=0, is_fwd=False, upd=True)
    #     schedule[1][-2] = Task(device_id=1, stage_id=1, mb_idx=0, is_fwd=False, upd=True)
    #     schedule[2][-3] = Task(device_id=2, stage_id=2, mb_idx=0, is_fwd=False, upd=True)
    #     schedule[3][-4] = Task(device_id=3, stage_id=3, mb_idx=0, is_fwd=False, upd=True)
    #     schedule[2][4] = schedule[2][6]
    #     schedule[2][6] = schedule[2][8]
    #     schedule[2][8] = None
    #     schedule[1][4] = schedule[1][7]
    #     schedule[1][7] = None
    # elif num_stages == 2:
    #     schedule[0][-1] = Task(device_id=0, stage_id=0, mb_idx=0, is_fwd=False, upd=True)
    #     schedule[1][-2] = Task(device_id=1, stage_id=1, mb_idx=0, is_fwd=False, upd=True)
    #     schedule[0][2] = schedule[0][4]
    #     schedule[0][4] = schedule[0][6]
    #     schedule[0][6] = None
    return schedule

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

def grid_schedule_to_DAG_schedule(
    grid_schedule: List[List[Task]],
    data_dependencies: List[Edge],
    all_stats: dict,
    comm_cost: int,
    num_mbs: int,
) -> DAG:
    """
    Convert a grid schedule to a DAG representation with temporal and data dependencies.
    
    Args:
        grid_schedule: 2D list where each row represents a device and each column represents a time step
        data_dependencies: List of edges defining data flow between stages
        all_stats: Dictionary containing timing and memory stats for each stage
        
    Returns:
        DAG object representing the schedule with temporal and data dependencies
    """
    # Helper function to get duration and memory for a task
    def get_task_stats(task: Task, all_stats: dict):
        if task.upd:
            stage_key = f"stage{task.stage_id}"
            return (
                all_stats[stage_key][f"Stage{task.stage_id}_Update"]["cuda_time_ms"],
                all_stats[stage_key][f"Stage{task.stage_id}_Update"]["cuda_mem_peak_GB"],
            )
        elif task.is_fwd:
            stage_key = f"stage{task.stage_id}"
            return (
                all_stats[stage_key][f"Stage{task.stage_id}_Forward"]["cuda_time_ms"],
                all_stats[stage_key][f"Stage{task.stage_id}_Forward"]["cuda_mem_peak_GB"],
            )
        else:
            stage_key = f"stage{task.stage_id}"
            return (
                all_stats[stage_key][f"Stage{task.stage_id}_Backward"]["cuda_time_ms"],
                all_stats[stage_key][f"Stage{task.stage_id}_Backward"]["cuda_mem_peak_GB"],
            )
    
    # validate schedule
    validate_schedule(grid_schedule, data_dependencies, num_mbs)
    
    # Create DAG nodes from all tasks in the grid schedule
    nodes = {}
    task_to_node_id = {}  # Map from Task objects to node IDs
    
    for device_idx, device_schedule in enumerate(grid_schedule):
        for time_step, task in enumerate(device_schedule):
            if task is not None:
                # Create unique node ID
                node_id = f"{'f' if task.is_fwd else 'b' if not task.upd else 'u'}_{task.device_id}_{task.stage_id}_{task.mb_idx}"
                if task.upd:
                    node_id = f"u_{task.stage_id}"
                
                # Get duration and memory stats
                duration, memory = get_task_stats(task, all_stats)
                
                # Create DAGTask
                dag_task = DAGTask(
                    fwd=task.is_fwd,
                    upd=task.upd,
                    device=task.device_id,
                    stage=task.stage_id,
                    mbtch=task.mb_idx,
                    duration=duration,
                    mem=memory,
                )
                
                nodes[node_id] = dag_task
                task_to_node_id[task] = node_id
    
    # Create DAG
    dag = DAG(grid_schedule, nodes, comm_cost=comm_cost)
    
    # Add temporal dependencies (adjacent tasks on same device)
    for device_idx, device_schedule in enumerate(grid_schedule):
        # Find all non-None tasks on this device
        non_none_tasks = [task for task in device_schedule if task is not None]
        
        # Create edges between consecutive tasks
        for i in range(len(non_none_tasks) - 1):
            current_task = non_none_tasks[i]
            next_task = non_none_tasks[i + 1]
            
            current_node_id = task_to_node_id[current_task]
            next_node_id = task_to_node_id[next_task]
            
            dag.add_edge(current_node_id, next_node_id)

    # Map each stage to its device (assumes consistent placement per stage)
    stage_to_device: Dict[int, int] = {}
    device_to_stages: Dict[int, Set[int]] = defaultdict(set)
    for node_id, dag_task in nodes.items():
        if dag_task.fwd:
            stage_to_device[dag_task.stage] = dag_task.device
            device_to_stages[dag_task.device].add(dag_task.stage)

    # Helper to compute node ids for forward/backward/update
    def fwd_node_id(stage: int, mb_idx: int) -> str:
        device = stage_to_device.get(stage)
        return f"f_{device}_{stage}_{mb_idx}" if device is not None else ""

    def bwd_node_id(stage: int, mb_idx: int) -> str:
        device = stage_to_device.get(stage)
        return f"b_{device}_{stage}_{mb_idx}" if device is not None else ""

    def upd_node_id(stage: int) -> str:
        return f"u_{stage}"

    # Add forward data dependencies across stages for each microbatch
    for dep in data_dependencies:
        for mb in range(num_mbs):
            src_id = fwd_node_id(dep.from_stage, mb)
            dst_id = fwd_node_id(dep.to_stage, mb)
            assert src_id in nodes and dst_id in nodes
            dag.add_edge(src_id, dst_id)

    # Add backward data dependencies in reverse stage order for each microbatch
    for dep in data_dependencies:
        for mb in range(num_mbs):
            src_id = bwd_node_id(dep.to_stage, mb)
            dst_id = bwd_node_id(dep.from_stage, mb)
            assert src_id in nodes and dst_id in nodes
            dag.add_edge(src_id, dst_id)

    # Add edge from final backward of final microbatch on each device to its update
    final_mb = num_mbs - 1
    for device, stages in device_to_stages.items():
        for stage in stages:
            b_final_id = bwd_node_id(stage, final_mb)
            u_id = upd_node_id(stage)
            assert b_final_id in nodes # and u_id in nodes
            dag.add_edge(b_final_id, u_id)
    
    return dag
