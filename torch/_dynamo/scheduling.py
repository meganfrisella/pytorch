from enum import Enum
from typing import NamedTuple
import ray
from ray.dag import InputNode, MultiOutputNode

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

    # maps mb_idx to the ref resulting from a forward call on the microbatch
    fwd_refs = dict()

    # maps mb_idx to a dict that maps stage_id to the refs output from the stage's backward on that microbatch
    bwd_ref_dicts = dict()

    # iterate over evrery task in the schedule
    for i in range(num_steps):
        for j in range(num_stages-1, -1, -1):
            task = schedule[j][i]
            if task:
                stage_id, mb_idx, is_fwd = task
                assert stage_id == j
                if is_fwd:
                    # if this is the first forward task for a microbatch, dispatch the forward task
                    # LIMITATION: forward for all stages is dispatched by this call, cannot interleave
                    # forward tasks with other tasks.
                    if mb_idx not in fwd_refs:
                        fwd_refs[mb_idx] = model(*params, dynamo_mb=mb_idx)
                else:
                    # log order of task dispatch by printing
                    # also see output_graph.py:1785 where we log forward dispatch
                    # print(f"Calling backward stage {stage_id} mb {mb_idx}")
                    num_bwd_targets = len(get_backward_targets(stage_id, dag_edges))
                    if mb_idx not in bwd_ref_dicts:
                        # if this is the first backward task for a microbatch, dispatch the
                        # backward task and cache the resulting ref(s)
                        fwd_ref = fwd_refs[mb_idx]
                        bwd_ref_dicts[mb_idx] = dict()
                        bwd_ref_dicts[mb_idx][stage_id] = (
                            actors[stage_id]
                            .backward.options(num_returns=num_bwd_targets*2)
                            .remote(
                                mb_idx, fwd_ref.get_ref(), loss_fn=loss_fn
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
                        bwd_ref_dicts[mb_idx][stage_id] = (
                            actors[stage_id]
                            .backward.options(num_returns=num_bwd_targets*2)
                            .remote(mb_idx, bwd_ref)
                        )

    # add update as the last column in the schedule
    # ASSUMPTION: the last row in the user-inputted schedule is a row of Nones
    # TOOD: consider making update explicit in the user-input schedule
    ret = []
    for stage_id in range(num_stages):
        num_bwd_targets = len(get_backward_targets(stage_id, dag_edges))
        done_refs = []
        for _, bwd_ref_dict in bwd_ref_dicts.items():
            bwd_refs = bwd_ref_dict[stage_id]
            bwd_refs = bwd_refs[num_bwd_targets:]
            if isinstance(bwd_refs, list):
                done_refs += bwd_refs
            else:
                done_refs.append(bwd_refs)
        upd = actors[stage_id].update.remote(*done_refs)
        ret.append(upd)
    return ret


# IGNORE the code below
# we shouldn't need to worry about schedule traversal order

# def execute_schedule_backwards(model, schedule, dag_edges: list[DAGEdge], params, truth, loss_fn):
#     assert (
#         hasattr(model, "_ray_actors")
#         and "model must be compiled with torch.compile with flag distribute=True"
#     )
#     actors = model._ray_actors
#     num_steps, num_stages = len(schedule[0]), len(schedule)
#     fwd_refs = dict()
#     bwd_ref_dicts = dict()
#     for i in range(num_steps):
#         for j in reversed(range(num_stages)):
#             task = schedule[j][i]
#             if task:
#                 stage_id, mb_idx, is_fwd = task
#                 if is_fwd:
#                     if mb_idx not in fwd_refs:
#                         fwd_refs[mb_idx] = model(*params, dynamo_mb=mb_idx)
#                 else:
#                     print(f"Calling backward stage {stage_id} mb {mb_idx}")
#                     num_bwd_targets = len(get_backward_targets(stage_id, dag_edges))
#                     if mb_idx not in bwd_ref_dicts:
#                         fwd_ref = fwd_refs[mb_idx]
#                         bwd_ref_dicts[mb_idx] = dict()
#                         bwd_ref_dicts[mb_idx][stage_id] = (
#                             actors[stage_id]
#                             .backward.options(num_returns=num_bwd_targets)
#                             .remote(
#                                 mb_idx, fwd_ref.get_ref(), truth=truth, loss_fn=loss_fn
#                             )
#                         )
#                     else:
#                         to_stage = [
#                             edge.to_stage
#                             for edge in dag_edges
#                             if edge.from_stage == stage_id
#                         ]
#                         assert len(to_stage) == 1
#                         to_stage = to_stage[0]
#                         targets = get_backward_targets(to_stage, dag_edges)
#                         idx = targets.index(DAGEdge(stage_id, to_stage))
#                         assert to_stage in bwd_ref_dicts[mb_idx]
#                         bwd_refs = bwd_ref_dicts[mb_idx][to_stage]
#                         bwd_ref = (
#                             bwd_refs
#                             if not isinstance(bwd_refs, list)
#                             else bwd_refs[idx]
#                         )
#                         bwd_ref_dicts[mb_idx][stage_id] = (
#                             actors[stage_id]
#                             .backward.options(num_returns=num_bwd_targets)
#                             .remote(mb_idx, bwd_refs)
#                         )
#     ret = []
#     for stage_id in range(num_stages):
#         done_refs = []
#         for _, bwd_ref_dict in bwd_ref_dicts.items():
#             bwd_refs = bwd_ref_dict[stage_id]
#             if isinstance(bwd_refs, list):
#                 done_refs += bwd_refs
#             else:
#                 done_refs.append(bwd_refs)
#         upd = actors[stage_id].update.remote(*done_refs)
#         ret.append(upd)
#     return ret
