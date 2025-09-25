from torch._dynamo.piper import Task, DAGEdge


def print_schedule(schedule):
    for stage in schedule:
        for step in stage:
            if step:
                string = f"{step.stage_id}:{step.mb_idx}:{'u' if step.upd else 'f' if step.is_fwd else 'b'}"
            else:
                string = " -- "
            print(string, end="\t")
        print()

def build_gpipe_schedule(n_mubatches: int, num_stages: int):
    steps = n_mubatches + num_stages - 1
    schedule = [[None] * (steps * 2 + 1) for _ in range(num_stages)]
    for step in range(steps):
        for stage_id in range(num_stages):
            mubatch_idx = step - stage_id
            if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                schedule[stage_id][step] = Task(stage_id, stage_id, mubatch_idx, True, False)

    for step in range(steps, steps * 2):
        for stage_id in reversed(range(num_stages)):
            mubatch_idx = (step - steps) - (num_stages - stage_id - 1)
            if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                schedule[stage_id][step] = Task(stage_id, stage_id, mubatch_idx, False, False)
    for i, stage in enumerate(range(num_stages)):
        schedule[stage][-i-1] = Task(stage_id=stage, device_id=stage, mb_idx=0, is_fwd=False, upd=True)
    return schedule

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