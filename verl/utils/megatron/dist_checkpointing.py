
from megatron.core import dist_checkpointing, mpu
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)


def save_dist_checkpointing(sharded_state_dict, ckpt_path, async_save=False):
    validate_sharding_integrity = True
    save_strategy = get_default_save_sharded_strategy("torch_dist")
    save_strategy = FullyParallelSaveStrategyWrapper(
        save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    async_save_request = dist_checkpointing.save(
        sharded_state_dict,
        ckpt_path,
        sharded_strategy=save_strategy,
        async_sharded_save=async_save,
        validate_access_integrity=validate_sharding_integrity,
    )

    return async_save_request


def load_dist_checkpointing(sharded_state_dict, ckpt_dir):
    load_strategy = get_default_load_sharded_strategy(ckpt_dir)
    load_strategy = FullyParallelLoadStrategyWrapper(
        load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    state_dict = dist_checkpointing.load(sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy)

    return state_dict
