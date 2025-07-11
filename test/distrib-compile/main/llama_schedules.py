from torch._dynamo.scheduling import Task, DAGEdge


def print_schedule(schedule):
    for stage in schedule:
        for step in stage:
            if step:
                string = f"{step.mb_idx}:{'f' if step.is_fwd else 'b'}"
            else:
                string = " - "
            print(string, end="\t")
        print()

def build_gpipe_schedule(n_mubatches: int, num_stages: int):
    steps = n_mubatches + num_stages - 1
    schedule = [[None] * (steps * 2 + 1) for _ in range(num_stages)]
    for step in range(steps):
        for stage_id in range(num_stages):
            mubatch_idx = step - stage_id
            if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                schedule[stage_id][step] = Task(stage_id, mubatch_idx, True)

    for step in range(steps, steps * 2):
        for stage_id in reversed(range(num_stages)):
            mubatch_idx = (step - steps) - (num_stages - stage_id - 1)
            if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                schedule[stage_id][step] = Task(stage_id, mubatch_idx, False)
    return schedule

def build_1f1b_schedule(n_mubatches: int, num_stages: int):
    steps = n_mubatches + num_stages - 1
    schedule = [[None] * (steps * 2) for _ in range(num_stages)]
    stage_mubatch = [[0, 0] for _ in range(num_stages)]
    for step in range(num_stages):
        for stage_id in range(num_stages):
            if step >= stage_id:
                mubatch_idx = stage_mubatch[stage_id][0]
                if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                    schedule[stage_id][step] = Task(
                        stage_id, mubatch_idx, True
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
                        stage_id, mubatch_idx, task_type
                    )
                    stage_mubatch[stage_id][fwd_or_bwd] += 1
    return schedule