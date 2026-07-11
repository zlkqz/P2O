"""Model and data parallel groups."""

import os
from typing import Optional

import sglang.srt.distributed.parallel_state as ps
import torch
import torch.distributed
from sglang.srt.distributed.parallel_state import (
    get_pp_group,
    get_world_group,
    init_distributed_environment,
    init_model_parallel_group,
)

"""
This version is strongly tied with Megatron to implement HybridEngine and weight sharing between vllm and Megatron.
- We assume the Megatron tp+dp+pp world is already established before calling this function.

"""

_DEVICE_MESH = None

_TP = None
_PP = None


def initialize_parallel_state(
    distributed_init_method: str = "env://",
    backend: str = "nccl",
    tensor_model_parallel_size: int = 1,
    num_tp_per_train_tp: int = 1,
    pipeline_model_parallel_size: int = 1,
):
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    rank = int(os.getenv("RANK", "-1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    world_size = int(os.getenv("WORLD_SIZE", "-1"))
    assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
    init_distributed_environment(world_size, rank, distributed_init_method, local_rank, backend)
    if torch.distributed.get_world_size() > 1:
        initialize_model_parallel_for_sglang(
            tensor_model_parallel_size=tensor_model_parallel_size,
            num_tensor_model_parallel_groups_per_train_tp=num_tp_per_train_tp,
        )
    else:
        initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, backend)


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, backend)
        return

    assert get_tensor_model_parallel_world_size() == tensor_model_parallel_size, (
        f"tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. {tensor_model_parallel_size=}"
    )
    pp_world_size = get_pp_group().world_size
    assert pp_world_size == pipeline_model_parallel_size, (
        f"pipeline parallel group already initialized, but of unexpected size: {pp_world_size=} vs. "
        f"{pipeline_model_parallel_size=}"
    )


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return _TP is not None


def initialize_model_parallel_for_sglang(
    tensor_model_parallel_size: int,
    num_tensor_model_parallel_groups_per_train_tp: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> None:
    pass

    assert torch.distributed.is_initialized()

    assert isinstance(tensor_model_parallel_size, int)


    assert ps._TP is None, "tensor model parallel group is already initialized"

    global _TP

    world_size: int = torch.distributed.get_world_size()

    backend = torch.distributed.get_backend()

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size

    if num_tensor_model_parallel_groups_per_train_tp == 1:
        assert _TP is None, "tensor model parallel group is already initialized"
        group_ranks = []
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            group_ranks.append(ranks)
        _TP = init_model_parallel_group(
            group_ranks=group_ranks,
            local_rank=get_world_group().local_rank,
            backend=backend,
            use_custom_allreduce=False,
            use_message_queue_broadcaster=True,
        )
        ps._TP = _TP
    else:

        train_tp = num_tensor_model_parallel_groups_per_train_tp * tensor_model_parallel_size
        assert _TP is None, "tensor model parallel group is already initialized"
        group_ranks = []
        for i in range(num_tensor_model_parallel_groups // num_tensor_model_parallel_groups_per_train_tp):
            start = train_tp * i
            end = train_tp * (i + 1)
            for j in range(num_tensor_model_parallel_groups_per_train_tp):
                ranks = list(range(start, end, num_tensor_model_parallel_groups_per_train_tp))
                for i in range(len(ranks)):
                    ranks[i] += j
                group_ranks.append(ranks)
        _TP = init_model_parallel_group(
            group_ranks=group_ranks,
            local_rank=get_world_group().local_rank,
            backend=backend,
            use_custom_allreduce=False,
            use_message_queue_broadcaster=True,
        )
        ps._TP = _TP



    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    _PP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, use_custom_allreduce=False)
    ps._PP = _PP


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    NOTE: This method is a hack from the open-sourced version without
    asertion of world_size = tp * pp

    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(ps.get_world_group().device_group)


    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size

    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size))
        group_ranks.append(ranks)

    if ps._TP is not None:
        _TP = ps._TP
    else:
        _TP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_custom_allreduce=False,
            use_message_queue_broadcaster=True,
        )
        ps._TP = _TP

    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    if ps._TP is not None:
        _PP = ps._TP
    else:
        _PP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, use_custom_allreduce=False)
        ps._PP = _PP


"""
Device mesh utilities
"""


def get_device_mesh():
    assert _DEVICE_MESH is not None, "device mesh is not initialized"
    return _DEVICE_MESH


"""
Tensor model parallel utilities
"""


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""

    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP.device_group


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size
