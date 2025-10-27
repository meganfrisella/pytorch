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
    steps = n_mubatches + num_stages - 2
    schedule = [[None] * (steps * 2 + 1) for _ in range(num_stages)]
    for step in range(steps):
        for stage_id in range(num_stages):
            if stage_id <= 1:
                mubatch_idx = step
            elif stage_id == 2:
                mubatch_idx = step - 1
            else:
                assert False and "CLIP must have 3 stages"
            if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                schedule[stage_id][step] = Task(stage_id, mubatch_idx, True)

    for step in range(steps, steps * 2):
        for stage_id in reversed(range(num_stages)):
            if stage_id <= 1:
                mubatch_idx = step - steps - 1
            elif stage_id == 2:
                mubatch_idx = step - steps
            else:
                assert False and "CLIP must have 3 stages"
            if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                schedule[stage_id][step] = Task(stage_id, mubatch_idx, False)
    return schedule
