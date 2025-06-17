from enum import Enum
from typing import NamedTuple


class Task(NamedTuple):
    stage_id: int
    mb_idx: int
    is_fwd: bool


class DAGEdge(NamedTuple):
    from_stage: int
    to_stage: int


def get_backward_targets(stage_id: int, dag_edges: list[DAGEdge]):
    return [edge for edge in dag_edges if edge.to_stage == stage_id]


def execute_schedule(model, schedule, dag_edges: list[DAGEdge], params, truth, loss_fn):
    assert (
        hasattr(model, "_ray_actors")
        and "model must be compiled with torch.compile with flag distribute=True"
    )
    actors = model._ray_actors
    num_steps, num_stages = len(schedule[0]), len(schedule)
    fwd_refs = dict()
    bwd_ref_dicts = dict()
    for i in range(num_steps):
        for j in range(num_stages):
            task = schedule[j][i]
            if task:
                stage_id, mb_idx, is_fwd = task
                # print(
                #     f"Processing stage {stage_id} mb {mb_idx} {'fwd' if is_fwd else 'bwd'}"
                # )
                if is_fwd:
                    if mb_idx not in fwd_refs:
                        fwd_refs[mb_idx] = model(*params, dynamo_mb=mb_idx)
                else:
                    # print(f"Calling backward stage {stage_id} mb {mb_idx}")
                    num_bwd_targets = len(get_backward_targets(stage_id, dag_edges))
                    if mb_idx not in bwd_ref_dicts:
                        fwd_ref = fwd_refs[mb_idx]
                        bwd_ref_dicts[mb_idx] = dict()
                        bwd_ref_dicts[mb_idx][stage_id] = (
                            actors[stage_id]
                            .backward.options(num_returns=num_bwd_targets)
                            .remote(
                                mb_idx, fwd_ref.get_ref(), truth=truth, loss_fn=loss_fn
                            )
                        )
                    else:
                        to_stage = [
                            edge.to_stage
                            for edge in dag_edges
                            if edge.from_stage == stage_id
                        ]
                        assert len(to_stage) == 1
                        to_stage = to_stage[0]
                        targets = get_backward_targets(to_stage, dag_edges)
                        idx = targets.index(DAGEdge(stage_id, to_stage))
                        assert to_stage in bwd_ref_dicts[mb_idx]
                        bwd_refs = bwd_ref_dicts[mb_idx][to_stage]
                        bwd_ref = (
                            bwd_refs
                            if not isinstance(bwd_refs, list)
                            else bwd_refs[idx]
                        )
                        bwd_ref_dicts[mb_idx][stage_id] = (
                            actors[stage_id]
                            .backward.options(num_returns=num_bwd_targets)
                            .remote(mb_idx, bwd_refs)
                        )
    ret = []
    for stage_id in range(num_stages):
        done_refs = []
        for _, bwd_ref_dict in bwd_ref_dicts.items():
            bwd_refs = bwd_ref_dict[stage_id]
            if isinstance(bwd_refs, list):
                done_refs += bwd_refs
            else:
                done_refs.append(bwd_refs)
        upd = actors[stage_id].update.remote(*done_refs)
        ret.append(upd)
    return ret
