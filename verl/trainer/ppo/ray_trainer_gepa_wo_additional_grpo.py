"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
import subprocess
import threading
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint, pformat
from typing import Any, Optional, Dict

import numpy as np
import ray
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
import random

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tracking import ValidationGenerationsLogger

from verl.workers.reward_manager.grouped_valid import GroupedValidRewardManager

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
else:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

GEPA_PROMPT_TEXT_KEY = "__gepa_prompt_text__"
NO_TEMPLATE_RAW_PROMPT_KEY = "__no_template_raw_prompt__"

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        print(f"Node available resources: {node_available_resources}")
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data.batch["response_mask"]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self._last_checkpoint_step: Optional[int] = None
        self._gepa_resume_info: Optional[dict] = None

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GPG,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self._gepa_cfg = self._init_gepa_cfg()
        self._gepa_epoch_template = ""
        self._current_epoch_hard_samples: list[Dict] = []
        self._gepa_epoch_uid_cache: set[str] = set()
        self._prev_epoch_hard_signatures: set[str] = set()
        self._prev_epoch_hard_signatures_train: set[str] = set()
        self._prev_epoch_hard_signatures_dev: set[str] = set()
        self._prev_epoch_hard_signatures_leftover: set[str] = set()
        self._gepa_epoch_candidates: list[str] = []
        self._gepa_epoch_dev_scores: list[float] = []
        self._gepa_epoch_training_templates: list[str] = []
        self._gepa_epoch_training_template_dev_scores: dict[str, float] = {}
        self._gepa_epoch_training_template_leftover_scores: dict[str, float] = {}
        self.is_gepa_loading_from_checkpoint = False
        self._logger = None

    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"} and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy in {"fsdp", "fsdp2"}:
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        total_training_steps += self.config.trainer.total_epochs - 1

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _init_gepa_cfg(self) -> dict[str, Any]:
        """Initialize GEPA-related runtime configuration with sensible defaults."""
        default_cfg = {
            "enabled": True,
            "add_near_hard": True,
            "hard_set_min_count": 100,
            "train_ratio": 150,
            "dev_ratio": 200,
            "reflection_minibatch_size": 3,
            "auto": None,
            "max_full_evals": None,
            "max_metric_calls": None,
            "select_top_k_template_to_train": False,
            "best_template_k": 3,
            "evaluate_pass_at_k_before_next_epoch": True,
            "gepa_select_pareto_k": 1,
            "reflection_max_tokens": 2048,
            "replace_prompts_when_log_prob": False,
            "fix_importance_ratio": True,
            "use_reference_model_to_reflect": False,
            "reflection_max_model_length": 32768,
            "reflection_temperature": None,
            "reflection_gpu_memory_utilization": 0.2,
            "thinking_in_reflection": False,
            "use_api_model_to_reflection": False,
            "api_base": "https://api.siliconflow.cn/v1",
            "model_name": "moonshotai/Kimi-K2-Instruct-0905",
            "api_key_file_name": "api_keys.txt",
            "proxies": None,
            "drop_failed_template_rollout": False,
            "left_1_no_template_rollout": False,
            "enable_template_ratio_mask": False,
            "template_ratio_cliprange_low": 0.2,
            "template_ratio_cliprange_high": 0.2,
            "soft_clip": False,
        }
        trainer_gepa_cfg = self.config["trainer"].get("gepa", None)
        if trainer_gepa_cfg is not None:
            try:
                user_cfg = OmegaConf.to_container(trainer_gepa_cfg, resolve=True)
                if isinstance(user_cfg, dict):
                    default_cfg.update(user_cfg)
            except Exception as exc:
                print(f"[GEPA] Warning: failed to parse trainer.gepa config ({exc}), using defaults.")
        self._validate_gepa_ratio(default_cfg["train_ratio"], "train_ratio", default_cfg["hard_set_min_count"])
        self._validate_gepa_ratio(default_cfg["dev_ratio"], "dev_ratio", default_cfg["hard_set_min_count"])
        self._validate_gepa_budget_cfg(default_cfg)
        self._validate_log_prob_config(
            default_cfg["replace_prompts_when_log_prob"],
            default_cfg["fix_importance_ratio"],
            default_cfg["enable_template_ratio_mask"]
        )

        if default_cfg.get("use_api_model_to_reflection", False) and default_cfg.get("use_reference_model_to_reflect", False):
            raise ValueError(
                "[GEPA] use_api_model_to_reflection and use_reference_model_to_reflect cannot both be True. "
                "Please set only one of them to True."
            )

        if default_cfg.get("use_api_model_to_reflection", False):
            if default_cfg.get("api_base") is None:
                raise ValueError(
                    "[GEPA] api_base must be specified when use_api_model_to_reflection is True."
                )
            if default_cfg.get("model_name") is None:
                raise ValueError(
                    "[GEPA] model_name must be specified when use_api_model_to_reflection is True."
                )

        if default_cfg.get("use_reference_model_to_reflect", False):
            reflection_gpu_util = default_cfg.get("reflection_gpu_memory_utilization", None)
            assert reflection_gpu_util is not None
            actor_gpu_util = float(self.config.actor_rollout_ref.rollout.gpu_memory_utilization)

            total_gpu_util = reflection_gpu_util + actor_gpu_util
            if total_gpu_util > 0.95:
                raise ValueError(
                    f"[GEPA] reflection_gpu_memory_utilization ({reflection_gpu_util}) + "
                    f"actor_rollout_ref.rollout.gpu_memory_utilization ({actor_gpu_util}) = "
                    f"{total_gpu_util} > 0.95. Please reduce one or both values."
                )

        self.auto_run_settings = {"light": 6, "medium": 12, "heavy": 18, "heavy_2": 36, "heavy_3": 54, "heavy_4": 72, "heavy_16": 288}
        if default_cfg["auto"] is not None:
            assert (
                default_cfg["auto"] in self.auto_run_settings
            ), f"Unknown GEPA auto setting {default_cfg['auto']}, expected one of {list(self.auto_run_settings.keys())}"

        return default_cfg

    @staticmethod
    def _validate_log_prob_config(replace_prompts_when_log_prob, fix_importance_ratio, enable_template_ratio_mask):
        true_count = sum([replace_prompts_when_log_prob, fix_importance_ratio, enable_template_ratio_mask])
        if true_count > 1:
            raise ValueError(
                "[GEPA] replace_prompts_when_log_prob、fix_importance_ratio和enable_template_ratio_mask"
                "只能有一个为True，当前有{}个为True".format(true_count)
            )

    @staticmethod
    def _validate_gepa_ratio(value, name: str, hard_set_min_count: int):
        if isinstance(value, int):
            if value < hard_set_min_count//2:
                raise ValueError(
                    f"[GEPA] {name} (int) must be >= hard_set_min_count//2, got {value}. "
                    "Please set trainer.gepa ratios to an integer >= hard_set_min_count//2 or a float <= 0.5."
                )
        elif isinstance(value, float):
            if not (0 < value <= 0.5):
                raise ValueError(
                    f"[GEPA] {name} (float) must be in (0, 0.3], got {value}. "
                    "Please set trainer.gepa ratios to an integer >= hard_set_min_count//2 or a float <= 0.5."
                )
        else:
            raise TypeError(
                f"[GEPA] {name} must be int or float, got {type(value)}. "
                "Please set trainer.gepa ratios to an integer >= hard_set_min_count//2 or a float <= 0.5."
            )

    @staticmethod
    def _validate_gepa_budget_cfg(cfg: dict[str, Any]):
        auto = cfg.get("auto")
        max_full = cfg.get("max_full_evals")
        max_calls = cfg.get("max_metric_calls")
        flags = [auto is not None, max_full is not None, max_calls is not None]
        if sum(flags) > 1:
            raise ValueError(
                "[GEPA] Only one of trainer.gepa.auto / max_full_evals / max_metric_calls can be set."
            )
        if sum(flags) == 0:
            cfg["auto"] = "heavy"

    def _gepa_enabled(self) -> bool:
        return bool(self._gepa_cfg.get("enabled", False))

    def _get_max_model_length(self) -> int:
        """Get the maximum model length from config or compute from prompt/response lengths.

        Returns:
            Maximum sequence length supported by the model.
        """
        if (hasattr(self, "_gepa_cfg") and
            (self._gepa_cfg.get("use_reference_model_to_reflect", False) or
             self._gepa_cfg.get("use_api_model_to_reflection", False)) and
            self._gepa_cfg.get("reflection_max_model_length") is not None):
            return int(self._gepa_cfg["reflection_max_model_length"])
        
        rollout_config = self.config.actor_rollout_ref.rollout
        if hasattr(rollout_config, "max_model_len") and rollout_config.max_model_len is not None:
            return int(rollout_config.max_model_len)
        
        actor_config = self.config.actor_rollout_ref.actor
        if hasattr(actor_config, "max_model_len") and actor_config.max_model_len is not None:
            return int(actor_config.max_model_len)
        
        if hasattr(self, "actor_rollout_wg") and self.actor_rollout_wg is not None:
            try:
                worker_config = getattr(self.actor_rollout_wg, "config", None)
                if worker_config and hasattr(worker_config, "max_model_len"):
                    max_len = worker_config.max_model_len
                    if max_len is not None:
                        return int(max_len)
            except Exception:
                pass
        
        prompt_length = self.config.data.get("max_prompt_length", 8192)
        response_length = self.config.data.get("max_response_length", 8192)
        computed_max = prompt_length + response_length
        
        return computed_max

    def _truncate_text_for_format_error(self, text: str, max_tokens: int = 100, tokenizer=None) -> str:
        """Truncate text for format error cases: keep head/tail max_tokens tokens with a marker.
        
        Args:
            text: Text to truncate.
            max_tokens: Number of tokens to keep from the beginning and the end (default: 100 each side).
            tokenizer: Tokenizer to use for counting tokens. If None, uses self.tokenizer.
        
        Returns:
            Truncated text with only first max_tokens, plus a note about format error.
        """
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens * 2:
            return text
        
        head_tokens = tokens[:max_tokens]
        tail_tokens = tokens[-max_tokens:]
        head_text = tokenizer.decode(head_tokens, skip_special_tokens=False).rstrip()
        tail_text = tokenizer.decode(tail_tokens, skip_special_tokens=False).lstrip()
        
        truncation_note = "\n\n[...中间部分已截断，格式错误...]\n\n"
        truncated = f"{head_text}{truncation_note}{tail_text}"
        
        truncated_tokens = len(tokenizer.encode(truncated, add_special_tokens=False))
        if truncated_tokens > max_tokens * 2 + 100:
            truncation_note = "\n\n[...已截断，格式错误]"
            truncated = f"{head_text}{truncation_note}{tail_text}"
        
        return truncated

    def _truncate_text_smart(self, text: str, max_tokens: int, tokenizer=None) -> str:
        """Intelligently truncate text to fit within max_tokens, preserving beginning and end.
        
        Args:
            text: Text to truncate.
            max_tokens: Maximum number of tokens allowed.
            tokenizer: Tokenizer to use for counting tokens. If None, uses self.tokenizer.
        
        Returns:
            Truncated text that fits within max_tokens.
        """
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
        
        truncation_marker = "\n\n[...中间部分已截断...]\n\n"
        marker_tokens = len(tokenizer.encode(truncation_marker, add_special_tokens=False))
        available_tokens = max_tokens - marker_tokens
        
        if available_tokens <= 0:
            return text[:100] + "..."
        
        begin_tokens = available_tokens * 40 // 100
        end_tokens = available_tokens * 40 // 100
        
        if begin_tokens + end_tokens + marker_tokens > max_tokens:
            excess = (begin_tokens + end_tokens + marker_tokens) - max_tokens
            begin_tokens = max(100, begin_tokens - excess // 2)
            end_tokens = max(100, end_tokens - (excess - excess // 2))
        
        begin_tokens_list = tokens[:begin_tokens]
        end_tokens_list = tokens[-end_tokens:]
        
        begin_text = tokenizer.decode(begin_tokens_list, skip_special_tokens=False).rstrip()
        end_text = tokenizer.decode(end_tokens_list, skip_special_tokens=False).lstrip()
        
        truncated = f"{begin_text}{truncation_marker}{end_text}"
        
        truncated_tokens = len(tokenizer.encode(truncated, add_special_tokens=False))
        if truncated_tokens > max_tokens * 1.1:
            return self._truncate_text_smart(text, max_tokens - 200, tokenizer)
        
        return truncated

    def _truncate_examples_to_fit(
        self, examples: list[dict[str, str | float]], max_total_tokens: int, template_text: str = ""
    ) -> list[dict[str, str | float]]:
        """Truncate examples to fit within max_total_tokens while preserving as much as possible.
        
        Args:
            examples: List of example dicts with keys like "Inputs", "Generated Outputs", "Feedback",
                     and metadata "format_score", "answer_score".
            max_total_tokens: Maximum total tokens for all examples plus template.
            template_text: The reflection prompt template text (excluding examples).
        
        Returns:
            Truncated examples list.
        """
        if not examples:
            return examples
        
        template_tokens = len(self.tokenizer.encode(template_text, add_special_tokens=False))
        available_tokens = max_total_tokens - template_tokens - 500
        
        if available_tokens <= 0:
            print(f"[GEPA] Warning: Template itself is too long ({template_tokens} tokens), cannot fit examples")
            return examples[:1]
        
        def _estimate_tokens(example_dict: dict[str, str | float]) -> int:
            input_text = str(example_dict.get("Inputs", ""))
            response = str(example_dict.get("Generated Outputs", ""))
            feedback_text = str(example_dict.get("Feedback", ""))
            example_text = f"{input_text}\n{response}\n{feedback_text}"
            return int(len(self.tokenizer.encode(example_text, add_special_tokens=False)) * 1.2)

        def _estimate_non_response_tokens(example_dict: dict[str, str | float]) -> int:
            input_text = str(example_dict.get("Inputs", ""))
            feedback_text = str(example_dict.get("Feedback", ""))
            return int(len(self.tokenizer.encode(f"{input_text}\n{feedback_text}", add_special_tokens=False)) * 1.2)

        minibatch_size = max(1, len(examples))
        threshold = max(1, available_tokens // minibatch_size)

        low_cost_examples: list[tuple[dict[str, str | float], int]] = []
        high_cost_examples: list[tuple[dict[str, str | float], int]] = []
        used_tokens = 0

        for example in examples:
            cost = _estimate_tokens(example)
            if cost <= threshold:
                low_cost_examples.append((example, cost))
                used_tokens += cost
            else:
                high_cost_examples.append((example, cost))

        truncated_examples: list[dict[str, str | float]] = [ex for ex, _ in low_cost_examples]
        remaining_tokens = available_tokens - used_tokens

        if remaining_tokens >= sum(cost for _, cost in high_cost_examples):
            truncated_examples.extend(ex for ex, _ in high_cost_examples)
            return truncated_examples

        format_error_high: list[tuple[dict[str, str | float], int]] = []
        normal_high: list[tuple[dict[str, str | float], int]] = []
        for example, cost in high_cost_examples:
            format_score = example.get("format_score", 1.0)
            if isinstance(format_score, (int, float)) and format_score == 0.0:
                format_error_high.append((example, cost))
            else:
                normal_high.append((example, cost))

        print(
            f"[GEPA] Truncating examples: threshold={threshold}, "
            f"{len(format_error_high)} format_error (high-cost), {len(normal_high)} normal (high-cost)"
        )

        for example, _ in format_error_high:
            response = str(example.get("Generated Outputs", ""))
            truncated_response = self._truncate_text_for_format_error(response, max_tokens=100)
            truncated_example = example.copy()
            truncated_example["Generated Outputs"] = truncated_response
            new_cost = _estimate_tokens(truncated_example)
            truncated_examples.append(truncated_example)
            remaining_tokens -= new_cost

        normal_total_cost = sum(cost for _, cost in normal_high)
        if remaining_tokens >= normal_total_cost:
            truncated_examples.extend(ex for ex, _ in normal_high)
            return truncated_examples

        num_normal = len(normal_high)
        if num_normal == 0:
            return truncated_examples

        tokens_per_example = max(100, remaining_tokens // num_normal) if remaining_tokens > 0 else 100
        for example, _ in normal_high:
            response = str(example.get("Generated Outputs", ""))
            non_response_tokens = _estimate_non_response_tokens(example)
            response_budget = max(50, tokens_per_example - non_response_tokens)
            truncated_response = self._truncate_text_smart(response, response_budget)
            truncated_example = example.copy()
            truncated_example["Generated Outputs"] = truncated_response
            truncated_examples.append(truncated_example)

        return truncated_examples

    def _reset_epoch_gepa_state(self):
        """Reset GEPA-related tracking containers at the beginning of every epoch."""
        self._current_epoch_hard_samples = []
        self._gepa_epoch_uid_cache = set()
        self._gepa_epoch_candidates = []
        self._gepa_epoch_dev_scores = []
        self._gepa_epoch_training_template_dev_scores = {}
        self._gepa_epoch_training_template_leftover_scores = {}

    def _extract_sample_from_batch_dict(
        self, batch_dict: dict[str, torch.Tensor | np.ndarray], index: int, uid: str | None = None
    ) -> dict[str, np.ndarray]:
        """Lightweight snapshot for a single sample that only keeps non-tensor fields.

        We intentionally avoid storing the `.batch` tensors from the training loop to
        reduce memory pressure. The stored object only contains the non-tensor
        (dtype=object) fields such as `raw_prompt`, `reward_model` metadata, etc.
        During GEPA we will rebuild a fresh ``DataProto`` with new tokenization
        using the tokenizer.
        """
        non_tensor_sample: dict[str, np.ndarray] = {}
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor):
                continue
            if isinstance(value, np.ndarray) and value.dtype != object:
                continue
            if key == "raw_prompt_ids":
                continue
            if key == "raw_prompt":
                raw_prompt_array_copy = np.empty(1, dtype=object)
                raw_prompt_array_copy[0] = deepcopy(value[index])
                non_tensor_sample[NO_TEMPLATE_RAW_PROMPT_KEY] = raw_prompt_array_copy

            if isinstance(value, np.ndarray):
                non_tensor_sample[key] = value[index:index+1]
            else:
                raise NotImplementedError(f"Find non np.ndarray type in batch_dict")

        prompt_text = self._decode_prompt_text_from_non_tensor_sample(non_tensor_sample)
        non_tensor_sample[GEPA_PROMPT_TEXT_KEY] = np.array([prompt_text], dtype=object)

        return non_tensor_sample

    def _decode_prompt_text_from_non_tensor_sample(self, non_tensor_sample: dict[str, Any]) -> str:
        """Extract user text from a lightweight non-tensor snapshot."""
        raw_prompt_arr = non_tensor_sample.get("raw_prompt")
        if isinstance(raw_prompt_arr, np.ndarray) and raw_prompt_arr.shape[0] >= 1:
            messages = list(deepcopy(raw_prompt_arr[0]))
            for message in reversed(messages):
                if message.get("role") == "user":
                    return str(message.get("content", "")).strip()
            raise RuntimeError("No user message found in raw_prompt.")
        raise RuntimeError("raw_prompt is required in non_tensor_sample to decode prompt text.")

    @staticmethod
    def _clone_dataproto(sample: DataProto) -> DataProto:
        """Deep copy helper for DataProto objects with batch size 1."""
        try:
            batch_clone = sample.batch.clone() if sample.batch is not None else None
            non_tensor_clone = {key: val.copy() for key, val in sample.non_tensor_batch.items()}
            meta_info_clone = deepcopy(sample.meta_info)
        except Exception as e:
            print(f"Error cloning dataproto: {e}")
            print(sample)

        return DataProto(batch=batch_clone, non_tensor_batch=non_tensor_clone, meta_info=meta_info_clone)
        

    def _encode_messages(self, messages: list[Dict], truncation="error", max_prompt_length=None, enable_thinking=False) -> tuple[list, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_prompt_str = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)
        model_inputs = self.tokenizer(raw_prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        if max_prompt_length is None:
            max_prompt_length = self.config.data.get("max_prompt_length", input_ids.shape[-1])
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation=truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        return model_inputs["input_ids"][0].tolist(), input_ids, attention_mask, position_ids


    def _apply_prompt_template_to_sample(self, sample: DataProto, prompt_template: str, truncation="error", max_prompt_length=None, enable_thinking=False) -> DataProto:
        """
        Return a cloned DataProto with prompt text replaced by the GEPA template.

        This operates on `raw_prompt`, which is expected to be present when
        `return_raw_chat=True` in the RL dataset config. Each element is a
        list of chat messages (dict with 'role' and 'content').
        """
        sample_clone = self._clone_dataproto(sample)
        assert (
            "raw_prompt" in sample_clone.non_tensor_batch
        ), "GEPA requires `return_raw_chat=True` so that `raw_prompt` is available."

        raw_prompt_arr = sample_clone.non_tensor_batch["raw_prompt"]
        assert (
            isinstance(raw_prompt_arr, np.ndarray) and raw_prompt_arr.shape[0] == 1
        ), f"Unexpected raw_prompt shape for GEPA: {type(raw_prompt_arr)}, {getattr(raw_prompt_arr, 'shape', None)}"

        messages = list(deepcopy(raw_prompt_arr[0]))
        assert isinstance(messages, list) and len(messages) >= 1, f"GEPA currently only supports chat-style prompts.\nRecieved Messages (Type: {type(messages)}   Length: {len(messages)}):\n{messages}"

        user_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                user_idx = idx
                break
        assert user_idx is not None, "Expected at least one user role message in raw_prompt."

        base_content = messages[user_idx].get("content", "")
        prompt_template = prompt_template or ""
        if prompt_template.strip():
            new_content = prompt_template.strip() + "\n\n" + base_content
        else:
            new_content = base_content
        messages[user_idx]["content"] = new_content

        raw_prompt_ids_list, input_ids, attention_mask, position_ids = self._encode_messages(messages, truncation, max_prompt_length, enable_thinking=enable_thinking)

        if sample_clone.batch is None:
            batch_size = (len(sample_clone),)
            sample_clone.batch = TensorDict(source={}, batch_size=batch_size, device=input_ids.device)
        sample_clone.batch["input_ids"] = input_ids
        sample_clone.batch["attention_mask"] = attention_mask
        sample_clone.batch["position_ids"] = position_ids

        sample_clone.non_tensor_batch["raw_prompt"] = np.array([messages], dtype=object)
        raw_prompt_ids_array = np.empty(1, dtype=object)
        raw_prompt_ids_array[0] = raw_prompt_ids_list
        sample_clone.non_tensor_batch["raw_prompt_ids"] = raw_prompt_ids_array
        sample_clone.non_tensor_batch[GEPA_PROMPT_TEXT_KEY] = np.array([new_content], dtype=object)
        return sample_clone

    def _build_gepa_batch(self, samples: list[DataProto], prompt_template: str) -> DataProto:
        """Construct a batched DataProto with the provided prompt template."""
        if len(samples) == 0:
            raise ValueError("Cannot build GEPA batch from empty sample list.")
        processed = [self._apply_prompt_template_to_sample(s, prompt_template, truncation="left") for s in samples]
        return DataProto.concat(processed)

    def _resolve_ratio_count(self, value: int | float, total_size: int) -> int:
        if isinstance(value, int):
            return min(value, total_size//2)
        ratio_count = max(self._gepa_cfg["hard_set_min_count"]//2, int(total_size * value))
        return min(ratio_count, total_size//2)

    def _split_gepa_dataset(
        self,
        samples: list[DataProto],
        train_ratio: int | float,
        dev_ratio: int | float,
        total_size: int,
    ) -> tuple[list[DataProto], list[DataProto], list[DataProto]]:
        """
        Shuffle and split GEPA samples into train/dev subsets.

        Args:
            samples: List of DataProto samples to split.
            train_dev_ratio: If int (>=hard_set_min_count//2), absolute dev count; if float (<=0.5), dev ratio.
            total_size: Total size of the hard set (used when ratio is float).

        Returns:
            (train_samples, dev_samples)
        """
        if len(samples) == 0:
            return [], [], []
        indices = list(range(len(samples)))
        random.shuffle(indices)

        dev_count = self._resolve_ratio_count(dev_ratio, total_size)
        dev_idxs = set(indices[:dev_count])
        dev = [samples[i] for i in dev_idxs]
        remaining_indices = [idx for idx in indices if idx not in dev_idxs]
        train_count = self._resolve_ratio_count(train_ratio, total_size)
        train_indices = remaining_indices[:train_count]
        leftover_indices = remaining_indices[train_count:]
        train_set = [samples[i] for i in train_indices] if train_indices else []
        leftover_set = [samples[i] for i in leftover_indices] if leftover_indices else []

        return train_set, dev, leftover_set

    def _evaluate_prompt_on_samples(
        self,
        samples: list[DataProto],
        prompt_template: str,
        rng: random.Random | None = None,
    ) -> list[dict]:
        """
        Evaluate a prompt template on a list of samples using the same reward
        computation logic as in `fit`, but without updating model parameters.

        Returns:
            List[dict]: one record per original sample containing:
                - 'reward': total reward (answer_score + format_score)
                - 'answer_score'
                - 'format_score'
                - 'golden_answer'
                - 'response'
        """
        if len(samples) == 0:
            return []

        rng = rng or random.Random(0)
        
        eval_batch_size = self.config.data.get("val_batch_size", None)
        if eval_batch_size is None:
            eval_batch_size = len(samples)

        all_records = []
        for start_idx in range(0, len(samples), eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]
            
            base_batch = self._build_gepa_batch(batch_samples, prompt_template)
            batch = RayPPOTrainer._clone_dataproto(base_batch)

            batch_keys_to_pop = [k for k in ["input_ids", "attention_mask", "position_ids"] if k in batch.batch.keys()]
            non_tensor_batch_keys_to_pop = [k for k in ["raw_prompt_ids", "multi_modal_data", "raw_prompt",
                                                        "tools_kwargs", "interaction_kwargs", "index", "agent_name"]
                                            if k in batch.non_tensor_batch]

            gen_batch = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)

            with torch.no_grad():
                if not self.async_rollout_mode:
                    gen_batch_output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                else:
                    gen_batch_output_padded = self.async_rollout_manager.generate_sequences(gen_batch_padded)

            gen_batch_output = unpad_dataproto(gen_batch_output_padded, pad_size=pad_size)

            batch = batch.union(gen_batch_output)
            if "response_mask" not in batch.batch.keys():
                batch.batch["response_mask"] = compute_response_mask(batch)

            reward_extra_infos_dict: dict[str, list] = {}
            with torch.no_grad():
                if self.use_rm:
                    reward_tensor_rm = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor_rm)

                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

            batch.batch["token_level_scores"] = reward_tensor
            if self.config.algorithm.use_kl_in_reward and self.use_reference_policy:
                with torch.no_grad():
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)
                    batch, _ = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            total_scores = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().tolist()
            answer_scores = reward_extra_infos_dict.get("answer_score", None)
            format_scores = reward_extra_infos_dict.get("format_score", None)
            assert answer_scores is not None and format_scores is not None, f"No detailed reward information returned"

            responses = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)

            golden_list = []
            rm_info = batch.non_tensor_batch.get("reward_model", None)
            if rm_info is not None:
                for i in range(len(total_scores)):
                    gt = rm_info[i].get("ground_truth", {})
                    target = gt.get("target", "")
                    golden_list.append(str(target))
            else:
                golden_list = ["" for _ in range(len(total_scores))]

            for i in range(len(total_scores)):
                rec = {
                    "reward": float(total_scores[i]),
                    "answer_score": float(answer_scores[i]) if i < len(answer_scores) else 0.0,
                    "format_score": float(format_scores[i]) if i < len(format_scores) else 0.0,
                    "golden_answer": golden_list[i],
                    "response": responses[i],
                }
                all_records.append(rec)
        
        return all_records

    def _evaluate_multiple_prompts_on_samples_batch(
        self,
        template_minibatch_pairs: list[tuple[str, list[DataProto]]],
        rng: random.Random | None = None,
    ) -> list[list[dict]]:
        """
        Batch evaluate multiple prompt templates on their respective minibatches.
        All templates are evaluated together in a single GPU batch for efficiency.
        
        Args:
            template_minibatch_pairs: List of (template, minibatch) tuples
            rng: Random number generator (optional)
            
        Returns:
            List of lists of records, one list per template-minibatch pair
        """
        if len(template_minibatch_pairs) == 0:
            return []
        
        rng = rng or random.Random(0)
        
        all_samples_with_metadata = []
        template_boundaries = []
        
        for template_idx, (template, minibatch) in enumerate(template_minibatch_pairs):
            start_idx = len(all_samples_with_metadata)
            for sample_idx, sample in enumerate(minibatch):
                all_samples_with_metadata.append((template_idx, sample_idx, sample))
            end_idx = len(all_samples_with_metadata)
            template_boundaries.append((start_idx, end_idx, template))
        
        if len(all_samples_with_metadata) == 0:
            return [[] for _ in template_minibatch_pairs]
        
        total_samples = len(all_samples_with_metadata)
        eval_batch_size = self.config.data.get("val_batch_size", None)
        if eval_batch_size is None:
            eval_batch_size = total_samples
        
        all_results = [[] for _ in template_minibatch_pairs]
        for start_idx in range(0, total_samples, eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, total_samples)
            batch_samples_with_metadata = all_samples_with_metadata[start_idx:end_idx]
            
            batch_template_boundaries = []
            current_template_idx = None
            batch_start = 0
            for local_idx, (template_idx, original_sample_idx, sample) in enumerate(batch_samples_with_metadata):
                if current_template_idx != template_idx:
                    if current_template_idx is not None:
                        batch_template_boundaries.append((batch_start, local_idx, template_boundaries[current_template_idx][2]))
                    batch_start = local_idx
                    current_template_idx = template_idx
            if current_template_idx is not None:
                batch_template_boundaries.append((batch_start, len(batch_samples_with_metadata), template_boundaries[current_template_idx][2]))
            
            all_batches = []
            for batch_start_idx, batch_end_idx, template in batch_template_boundaries:
                samples_for_template = [batch_samples_with_metadata[i][2] for i in range(batch_start_idx, batch_end_idx)]
                if len(samples_for_template) > 0:
                    base_batch = self._build_gepa_batch(samples_for_template, template)
                    batch = RayPPOTrainer._clone_dataproto(base_batch)
                    all_batches.append(batch)
            
            if len(all_batches) == 0:
                continue
            
            merged_batch = DataProto.concat(all_batches)
            
            batch_keys_to_pop = [k for k in ["input_ids", "attention_mask", "position_ids"] if k in merged_batch.batch.keys()]
            non_tensor_batch_keys_to_pop = [k for k in ["raw_prompt_ids", "multi_modal_data", "raw_prompt",
                                                        "tools_kwargs", "interaction_kwargs", "index", "agent_name"]
                                            if k in merged_batch.non_tensor_batch]
            
            gen_batch = merged_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }        
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
            
            with torch.no_grad():
                if not self.async_rollout_mode:
                    gen_batch_output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                else:
                    gen_batch_output_padded = self.async_rollout_manager.generate_sequences(gen_batch_padded)
            
            gen_batch_output = unpad_dataproto(gen_batch_output_padded, pad_size=pad_size)
            merged_batch = merged_batch.union(gen_batch_output)
            if "response_mask" not in merged_batch.batch.keys():
                merged_batch.batch["response_mask"] = compute_response_mask(merged_batch)
            
            reward_extra_infos_dict: dict[str, list] = {}
            with torch.no_grad():
                if self.use_rm:
                    reward_tensor_rm = self.rm_wg.compute_rm_score(merged_batch)
                    merged_batch = merged_batch.union(reward_tensor_rm)
                
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(data=merged_batch, reward_fn=self.reward_fn)
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(merged_batch, self.reward_fn)
            
            merged_batch.batch["token_level_scores"] = reward_tensor
            if self.config.algorithm.use_kl_in_reward and self.use_reference_policy:
                with torch.no_grad():
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(merged_batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(merged_batch)
                    merged_batch = merged_batch.union(ref_log_prob)
                    merged_batch, _ = apply_kl_penalty(
                        merged_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
            else:
                merged_batch.batch["token_level_rewards"] = merged_batch.batch["token_level_scores"]
            
            total_scores = merged_batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().tolist()
            answer_scores = reward_extra_infos_dict.get("answer_score", None)
            format_scores = reward_extra_infos_dict.get("format_score", None)
            assert answer_scores is not None and format_scores is not None, f"No detailed reward information returned"
            
            responses = self.tokenizer.batch_decode(merged_batch.batch["responses"], skip_special_tokens=True)
            
            golden_list = []
            rm_info = merged_batch.non_tensor_batch.get("reward_model", None)
            if rm_info is not None:
                for i in range(len(total_scores)):
                    gt = rm_info[i].get("ground_truth", {})
                    target = gt.get("target", "")
                    golden_list.append(str(target))
            else:
                golden_list = ["" for _ in range(len(total_scores))]
            
            batch_current_global_idx = 0
            for batch_start_idx, batch_end_idx, _ in batch_template_boundaries:
                template_idx = batch_samples_with_metadata[batch_start_idx][0]
                num_samples = batch_end_idx - batch_start_idx
                for i in range(num_samples):
                    local_idx = batch_current_global_idx + i
                    if local_idx < len(total_scores):
                        rec = {
                            "reward": float(total_scores[local_idx]),
                            "answer_score": float(answer_scores[local_idx]) if local_idx < len(answer_scores) else 0.0,
                            "format_score": float(format_scores[local_idx]) if local_idx < len(format_scores) else 0.0,
                            "golden_answer": golden_list[local_idx],
                            "response": responses[local_idx],
                        }
                        all_results[template_idx].append(rec)
                batch_current_global_idx += num_samples
        
        return all_results

    def _build_feedback_text(
        self,
        answer_score: float,
        format_score: float,
        golden_answer: str,
    ) -> str:
        """
        Build feedback text following the logic in start_for_reasoning.metric_with_feedback:

        - if format_score == 0: use lines 96-104 (format error)
        - elif answer_score == 0: use lines 112-113 (incorrect answer)
        - else: use lines 109-111 (correct answer)
        Additionally, append a line with Reward(answer_score).
        """
        golden_answer = str(golden_answer).strip()

        if format_score == 0:
            feedback_text = (
                "You have to provide final answer, without detailed explanations.\nIt is possible that you failed to output in the required format, either omitting the specified tag or including multiple ones.\n"
                f"The correct answer is '{golden_answer}'."
                "\n\nPlease analyze your rasoning trajectory with the correct answer, and check your output format."
            )
        elif answer_score == 0:
            feedback_text = f"Your final answer is incorrect. The correct answer is '{golden_answer}'."
        else:
            feedback_text = f"Your final answer is correct. The correct answer is '{golden_answer}'."

        feedback_text += f"\n\nReward (answer_score) = {answer_score:.2f}"
        return feedback_text

    def _compute_pass_at_k_with_rollout(
        self,
        samples: list[DataProto],
        template: str,
        k: int,
        rng: random.Random | None = None,
    ) -> float:
        """
        Compute pass@k by evaluating with k rollouts per sample.
        Uses _evaluate_prompt_on_samples for evaluation logic, then computes pass@k.
        
        Args:
            samples: List of DataProto samples to evaluate
            template: Prompt template to use
            k: Number of rollouts per sample (pass@k)
            rng: Random number generator (optional)
            
        Returns:
            pass@k score (float between 0 and 1)
        """

        def estimate_pass_at_k(n, c, k):
            """
            计算无偏 pass@k (即图片中的第一行公式)
            n: 采样的样本总数 (total_generated)
            c: 通过测试的样本数 (correct_samples)
            k: 想评估的 k 值 (pass@k)
            """
            if n < k:
                return -1.0 
            if c == 0:
                return 0.0
            prob_all_wrong = 1.0
            for i in range(k):
                prob_all_wrong *= (n - c - i) / (n - i)
            return 1.0 - prob_all_wrong

        if len(samples) == 0:
            return 0.0

        double_k = 2 * k
        
        rng = rng or random.Random(0)
        reward_epsilon = 1e-6
        
        all_sample_rewards = []
        repeated_samples = []
        for sample in samples:
            for _ in range(double_k):
                repeated_samples.append(sample)
        
        all_records = self._evaluate_prompt_on_samples(repeated_samples, template, rng)
        
        num_original_samples = len(samples)
        for i in range(num_original_samples):
            sample_rewards = [
                float(all_records[i * double_k + j].get("reward", 0.0))
                for j in range(double_k)
            ]
            all_sample_rewards.append(sample_rewards)
        
        pass_at_k = 0.
        for sample_rewards in all_sample_rewards:
            correct_num = sum(r > reward_epsilon for r in sample_rewards)
            pass_at_k += estimate_pass_at_k(double_k, correct_num, k)
        
        pass_at_k = pass_at_k / len(all_sample_rewards) if all_sample_rewards else 0.0
        return pass_at_k

    def _extract_input_text_from_sample(self, sample: DataProto) -> str:
        """Return the user-visible prompt text that produced the rollout."""
        raw_prompt_arr = sample.non_tensor_batch.get("raw_prompt")
        if isinstance(raw_prompt_arr, np.ndarray) and raw_prompt_arr.shape[0] == 1:
            messages = list(raw_prompt_arr[0])
            if isinstance(messages, list):
                for message in messages:
                    if message.get("role") == "user":
                        return str(message.get("content", "")).strip()
        prompt_arr = sample.non_tensor_batch.get(GEPA_PROMPT_TEXT_KEY)
        if isinstance(prompt_arr, np.ndarray) and len(prompt_arr) > 0:
            return str(prompt_arr[0]).strip()
        raise RuntimeError("No prompt text found in the sample.")

    def _attach_inputs_to_records(self, samples: list[DataProto], reward_records: list[dict]):
        """Annotate reward records with their originating input text."""
        for sample, record in zip(samples, reward_records, strict=True):
            record.setdefault("input_text", self._extract_input_text_from_sample(sample))

    def _record_to_instruction_example(self, record: dict) -> dict[str, str | float]:
        """Convert a rollout record into the InstructionProposal example schema.
        
        Returns:
            Dict with keys: "Inputs", "Generated Outputs", "Feedback", 
            and metadata keys: "format_score", "answer_score" for truncation logic.
        """
        feedback = self._build_feedback_text(
            record.get("answer_score", 0.0),
            record.get("format_score", 0.0),
            record.get("golden_answer", ""),
        )
        return {
            "Inputs": record.get("input_text", ""),
            "Generated Outputs": record.get("response", ""),
            "Feedback": feedback,
            "format_score": float(record.get("format_score", 0.0)),
            "answer_score": float(record.get("answer_score", 0.0)),
        }

    def _format_instruction_examples(self, examples: list[dict[str, str | float]]) -> str:
        """Render dataset_with_feedback into markdown, mirroring GEPA templates.
        
        Metadata fields (format_score, answer_score) are excluded from the formatted output.
        """
        metadata_keys = {"format_score", "answer_score"}
        
        def render_value(value, level=3):
            if isinstance(value, dict):
                lines = []
                for k, v in value.items():
                    lines.append(f"{'#' * level} {k}")
                    lines.append(render_value(v, min(level + 1, 6)))
                return "\n".join(lines)
            if isinstance(value, (list, tuple)):
                lines = []
                for idx, item in enumerate(value):
                    lines.append(f"{'#' * level} Item {idx + 1}")
                    lines.append(render_value(item, min(level + 1, 6)))
                return "\n".join(lines)
            return f"{str(value).strip()}\n"

        def convert(idx, sample_dict):
            sections = [f"# Example {idx + 1}"]
            for key, val in sample_dict.items():
                if key in metadata_keys:
                    continue
                sections.append(f"## {key}")
                sections.append(render_value(val, level=3))
            return "\n".join(sections)

        return "\n\n".join(convert(i, sample) for i, sample in enumerate(examples))

    def _build_reflection_prompt(self, template: str, dataset_examples: list[dict[str, str | float]]) -> str:
        """Construct InstructionProposal prompt identical to GEPA.
        
        This method automatically truncates examples if they exceed the model's maximum length.
        Examples may contain metadata fields (format_score, answer_score) which are used for
        truncation logic but excluded from the formatted output.
        """
        template_text = template.strip()
        escaped_template_text = template_text.replace("{", "{{").replace("}", "}}")
        
        template_part = (
            "I provided an assistant with the following instructions to perform a task for me:\n"
            "```\n"
            f"{escaped_template_text}\n"
            "```\n\n"
            "The following are examples of different task inputs provided to the assistant along with "
            "the assistant's response for each of them, and some feedback on how the assistant's "
            "response could be better:\n"
            "```\n"
            "{formatted_examples}\n"
            "```\n\n"
            "Your task is to write a new instruction for the assistant.\n\n"
            "Read the inputs carefully and identify the input format and infer detailed task "
            "description about the task I wish to solve with the assistant.\n\n"
            "Read all the assistant responses and the corresponding feedback. Identify all niche "
            "and domain specific factual information about the task and include it in the instruction, "
            "as a lot of it may not be available to the assistant in the future. The assistant may have "
            "utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.\n\n"
            "Provide the new instructions within ``` blocks."
        )
        
        max_model_len = self._get_max_model_length()
        
        max_prompt_tokens = max_model_len - self._gepa_cfg.get("reflection_max_tokens", 2048)
        
        truncated_examples = self._truncate_examples_to_fit(
            dataset_examples, 
            max_total_tokens=max_prompt_tokens,
            template_text=template_part.format(formatted_examples="")
        )
        
        formatted_examples = self._format_instruction_examples(truncated_examples)
        
        full_prompt = template_part.format(formatted_examples=formatted_examples)
        
        prompt_tokens = len(self.tokenizer.encode(full_prompt, add_special_tokens=False))
        if prompt_tokens > max_prompt_tokens:
            print(
                f"[GEPA] Warning: Reflection prompt is too long ({prompt_tokens} tokens > {max_prompt_tokens}). "
                f"Applying more aggressive truncation."
            )
            while prompt_tokens > max_prompt_tokens and len(truncated_examples) > 1:
                truncated_examples = truncated_examples[:-1]
                formatted_examples = self._format_instruction_examples(truncated_examples)
                full_prompt = template_part.format(formatted_examples=formatted_examples)
                prompt_tokens = len(self.tokenizer.encode(full_prompt, add_special_tokens=False))
                print(f"[GEPA] Reduced to {len(truncated_examples)} examples, prompt length: {prompt_tokens} tokens")
        
        return full_prompt

    def _create_vllm_llm(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.2,
        max_model_len: Optional[int] = None,
    ):
        """创建 vLLM LLM 实例
        
        Args:
            model_path: 模型路径
            tensor_parallel_size: 张量并行大小
            gpu_memory_utilization: GPU 内存利用率
            max_model_len: 最大模型长度
            
        Returns:
            vllm.LLM: LLM 实例
        """
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["VLLM_USE_RAY"] = "0"
        tmp = os.environ["CUDA_VISIBLE_DEVICES"]
        if self.config.original_cuda_visible_devices is not None:   
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.original_cuda_visible_devices
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES")
        from vllm import LLM
        
        print(f"[GEPA] Creating vLLM LLM instance (bypassing Ray, using local GPUs) "
              f"with model_path={model_path}, "
              f"tensor_parallel_size={tensor_parallel_size}, "
              f"gpu_memory_utilization={gpu_memory_utilization}, "
              f"max_model_len={max_model_len}, "
              f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "distributed_executor_backend": "mp",
        }
        
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        
        try:
            llm = LLM(**llm_kwargs)
            print(f"[GEPA] vLLM LLM instance created successfully")
            os.environ["CUDA_VISIBLE_DEVICES"] = tmp
            return llm
        except Exception as e:
            error_msg = f"Failed to create vLLM LLM instance: {e}"
            print(f"[GEPA] Error: {error_msg}")
            raise RuntimeError(error_msg)
    
    def _generate_with_vllm_llm(
        self,
        llm,
        prompts: list[str],
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        """使用 vLLM LLM 实例批量生成
        
        Args:
            llm: vLLM LLM 实例
            prompts: 输入提示列表
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            
        Returns:
            生成的文本列表
        """
        from vllm.sampling_params import SamplingParams
        
        if len(prompts) == 0:
            return []
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        try:
            outputs = llm.generate(prompts, sampling_params)
            
            generated_texts = []
            for output in outputs:
                if len(output.outputs) > 0:
                    generated_text = output.outputs[0].text
                    generated_texts.append(generated_text)
                else:
                    generated_texts.append("")
            
            return generated_texts
        except Exception as e:
            error_msg = f"Failed to generate with vLLM LLM: {e}"
            print(f"[GEPA] Error: {error_msg}")
            raise RuntimeError(error_msg)

    def _sample_instructions_from_prompts_batch(
        self,
        instruction_prompts: list[str],
    ) -> list[str]:
        """
        Batch generate improved templates from multiple reflection prompts.
        All prompts are processed together in a single GPU batch for efficiency.

        Args:
            instruction_prompts: List of reflection prompts

        Returns:
            List of improved templates (one per prompt)
        """
        if len(instruction_prompts) == 0:
            return []

        use_api_reflection = self._gepa_cfg.get("use_api_model_to_reflection", False)

        if use_api_reflection:
            print("[GEPA] Using API model for reflection...")

            from utils.openaihandler import load_api_keys_from_file, OpenAIHandler

            api_keys_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self._gepa_cfg.get("api_key_file_name", "api_keys.txt")
            )

            try:
                api_keys = load_api_keys_from_file(api_keys_file)
            except Exception as e:
                raise RuntimeError(
                    f"[GEPA] Failed to load API keys from {api_keys_file}: {e}"
                )

            api_base = self._gepa_cfg.get("api_base")
            model_name = self._gepa_cfg.get("model_name")
            proxies = self._gepa_cfg.get("proxies")

            handler = OpenAIHandler(
                api_keys=api_keys,
                api_base=api_base,
                model_name=model_name,
                max_workers=64,
                proxies=proxies,
            )

            temperature = self._gepa_cfg.get("reflection_temperature", None)
            if temperature is None:
                if hasattr(self.config.actor_rollout_ref.rollout, "temperature"):
                    temperature = self.config.actor_rollout_ref.rollout.temperature
                else:
                    temperature = 0.6

            max_tokens = self._gepa_cfg.get("reflection_max_tokens", 2048)

            print(f"[GEPA] Calling API with {len(instruction_prompts)} prompts...")
            generated_texts = handler.batch_generate(
                prompts=instruction_prompts,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=self._gepa_cfg.get("thinking_in_reflection", False),
            )

            improved_templates = []
            for text in generated_texts:
                text = text.strip()
                extracted_text = ""
                if text.count("```") >= 2:
                    start = text.find("```")
                    end = text.rfind("```")
                    if start >= 0 and end > start:
                        extracted_text = text[start + 3 : end].strip()
                    else:
                        extracted_text = text.strip("` \n")
                else:
                    extracted_text = text.strip("` \n")

                if extracted_text.lower().startswith("markdown"):
                    extracted_text = extracted_text[8:].strip()

                improved_templates.append(extracted_text)

            return improved_templates

        use_vllm_reflection = (
            hasattr(self, "_reflection_llm") and
            self._reflection_llm is not None and
            self._gepa_cfg.get("use_reference_model_to_reflect", False)
        )
        
        if use_vllm_reflection:
            improved_templates = []
            max_response_length = self._gepa_cfg.get("reflection_max_tokens", 2048)
            
            temperature = self._gepa_cfg.get("reflection_temperature", None)
            if temperature is None:
                if hasattr(self.config.actor_rollout_ref.rollout, "temperature"):
                    temperature = self.config.actor_rollout_ref.rollout.temperature
                else:
                    temperature = 0.6
            
            formatted_prompts = []
            for prompt in instruction_prompts:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=True, 
                        tokenize=False,
                        enable_thinking=self._gepa_cfg.get("thinking_in_reflection", False)
                    )
                    formatted_prompts.append(formatted_prompt)
                except Exception:
                    formatted_prompts.append(prompt)
            
            generated_texts = self._generate_with_vllm_llm(
                llm=self._reflection_llm,
                prompts=formatted_prompts,
                temperature=temperature,
                max_tokens=max_response_length,
            )
            
            for text in generated_texts:
                text = text.strip()
                extracted_text = ""
                if text.count("```") >= 2:
                    start = text.find("```")
                    end = text.rfind("```")
                    if start >= 0 and end > start:
                        extracted_text = text[start + 3 : end].strip()
                    else:
                        extracted_text = text.strip("` \n")
                else:
                    extracted_text = text.strip("` \n")

                if extracted_text.lower().startswith("markdown"):
                    extracted_text = extracted_text[8:].strip()

                improved_templates.append(extracted_text)

            return improved_templates
        
        eval_batch_size = self.config.data.get("val_batch_size", None)
        if eval_batch_size is None:
            eval_batch_size = len(instruction_prompts)
        
        improved_templates = []
        max_response_length = self._gepa_cfg.get("reflection_max_tokens", 2048)
        max_prompt_length = self._get_max_model_length() - max_response_length
        
        for start_idx in range(0, len(instruction_prompts), eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, len(instruction_prompts))
            batch_prompts = instruction_prompts[start_idx:end_idx]
            
            all_batches = []
            for prompt in batch_prompts:
                messages = [{"role": "user", "content": prompt}]
                sample = DataProto.from_single_dict({"raw_prompt": np.array([messages])})
                sample = self._apply_prompt_template_to_sample(
                    sample, "", truncation="left", 
                    max_prompt_length=max_prompt_length, 
                    enable_thinking=self._gepa_cfg.get("thinking_in_reflection", False)
                )
                all_batches.append(sample)
            
            merged_batch = DataProto.concat(all_batches)
            
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in merged_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in merged_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in merged_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in merged_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "index" in merged_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("index")
            if "agent_name" in merged_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")

            gen_batch = merged_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=non_tensor_batch_keys_to_pop)
            gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
                "response_length": max_response_length,
            }
            
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)

            with torch.no_grad():
                if not self.async_rollout_mode:
                    gen_output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                else:
                    gen_output_padded = self.async_rollout_manager.generate_sequences(gen_batch_padded)
            
            gen_output = unpad_dataproto(gen_output_padded, pad_size=pad_size)
            batch = gen_output
            responses = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            
            for idx in range(len(batch_prompts)):
                text = responses[idx].strip()
                extracted_text = ""
                if text.count("```") >= 2:
                    start = text.find("```")
                    end = text.rfind("```")
                    if start >= 0 and end > start:
                        extracted_text = text[start + 3 : end].strip()
                    else:
                        extracted_text = text.strip("` \n")
                else:
                    extracted_text = text.strip("` \n")

                if extracted_text.lower().startswith("markdown"):
                    extracted_text = extracted_text[8:].strip()

                improved_templates.append(extracted_text)

        return improved_templates

    def _rebuild_batch_input(self, batch: DataProto):
        """
        将batch中的prompts部分替换为未加prompt_template的版本

        Args:
            batch: 要处理的batch（已经过repeat，长度为base_batch_size * rollout.n）
        """
        assert (
            NO_TEMPLATE_RAW_PROMPT_KEY in batch.non_tensor_batch
        ), f"[GEPA] No '{NO_TEMPLATE_RAW_PROMPT_KEY}' key in batch.non_tensor_batch"

        max_prompt_length = self.config.data.get("max_prompt_length", batch.batch["input_ids"].shape[-1])
        truncation = self.config.data.get("truncation", "error")

        batch.non_tensor_batch["raw_prompt"] = batch.non_tensor_batch[NO_TEMPLATE_RAW_PROMPT_KEY]

        if "__gepa_template_applied__" in batch.non_tensor_batch:
            template_applied_mask = batch.non_tensor_batch["__gepa_template_applied__"]
            template_applied_indices = np.where(template_applied_mask)[0]

            if len(template_applied_indices) > 0:
                for idx in template_applied_indices:
                    messages = batch.non_tensor_batch["raw_prompt"][idx]
                    _, input_ids, attention_mask, _ = self._encode_messages(messages, truncation, max_prompt_length)

                    batch.batch["prompts"][idx] = input_ids.squeeze(0)
                    batch.batch["input_ids"][idx, :max_prompt_length] = input_ids.squeeze(0)
                    batch.batch["attention_mask"][idx, :max_prompt_length] = attention_mask.squeeze(0)
        else:
            batch_input_ids, batch_attention_mask = [], []
            for batch_idx in range(batch.non_tensor_batch["raw_prompt"].shape[0]):
                messages = batch.non_tensor_batch["raw_prompt"][batch_idx]
                _, input_ids, attention_mask, _ = self._encode_messages(messages, truncation, max_prompt_length)
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
            batch_input_ids = torch.cat(batch_input_ids, dim=0)
            batch_attention_mask = torch.cat(batch_attention_mask, dim=0)

            batch.batch["prompts"] = batch_input_ids
            batch.batch["input_ids"][:, :max_prompt_length] = batch_input_ids
            batch.batch["attention_mask"][:, :max_prompt_length] = batch_attention_mask

        batch.batch["position_ids"] = compute_position_id_with_mask(batch.batch["attention_mask"])

        return batch

    def _collect_very_hard_samples(
        self,
        batch: DataProto,
        uid_to_sample: dict[str, dict[str, np.ndarray]],
        rollouts_per_sample: int,
        reward_epsilon: float = 1e-6,
    ):
        """
        Identify "very hard" prompts (all-zero rewards) and optionally "near-hard" prompts
        (exactly one non-zero reward across rollouts).

        If `add_near_hard=True` in config, near-hard samples are directly added to hard set.
        """
        if batch.batch is None or "token_level_rewards" not in batch.batch:
            raise RuntimeError(
                "Token level rewards not found in the batch."
            )
        add_near_hard = self._gepa_cfg.get("add_near_hard", True)
        rewards = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().tolist()
        uids = batch.non_tensor_batch.get("uid", [])
        aggregated: dict[str, list[float]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            aggregated[uid].append(rewards[idx])
        for uid, reward_list in aggregated.items():
            if uid in self._gepa_epoch_uid_cache:
                continue

            zero_mask = [abs(r) <= reward_epsilon for r in reward_list]
            zero_count = sum(1 for v in zero_mask if v)
            non_zero_count = len(reward_list) - zero_count

            sample = uid_to_sample.get(uid)
            if sample is None:
                continue

            if non_zero_count == 0:
                self._current_epoch_hard_samples.append(sample)
                self._gepa_epoch_uid_cache.add(uid)
            elif add_near_hard and non_zero_count == 1 and zero_count >= 1:
                self._current_epoch_hard_samples.append(sample)
                self._gepa_epoch_uid_cache.add(uid)


    def _gepa_prompt_signature_from_messages(self, messages: list[dict]) -> str:
        """
        Build a deterministic signature string for a chat-style `messages` list.

        This is used to:
          - identify which samples were "hard" in the previous epoch (based on
            their raw chat messages),
          - and then, in the next epoch, detect the same samples in the GRPO
            training loop in order to inject the GEPA-optimized template.
        """
        import json

        return json.dumps(messages, ensure_ascii=False, sort_keys=True)

    def _compute_signatures_from_samples(self, samples: list[DataProto]) -> set[str]:
        """
        Compute signatures for a list of DataProto samples.

        Args:
            samples: List of DataProto samples

        Returns:
            Set of signature strings
        """
        signatures: set[str] = set()
        for sample_idx, sample in enumerate(samples):
            try:
                raw_prompt_arr = sample.non_tensor_batch.get("raw_prompt", None)
                if raw_prompt_arr is None:
                    continue
                messages = list(deepcopy(raw_prompt_arr[0]))
                if not isinstance(messages, list):
                    continue
                sig = self._gepa_prompt_signature_from_messages(messages)
                signatures.add(sig)
            except Exception as exc:
                print(f"[GEPA] Warning: failed to compute signature for sample idx={sample_idx}: {exc}")
        return signatures

    def _compute_prev_epoch_hard_signatures(self) -> set[str]:
        """
        Compute signatures for all hard samples collected in the *current* epoch.

        This is called at the end of an epoch (inside
        `_finalize_epoch_prompt_optimization`) and the resulting set is then
        used by the *next* epoch to decide which batch samples should receive
        GEPA template injection.
        """
        signatures: set[str] = set()
        for sample_idx, non_tensor_sample in enumerate(self._current_epoch_hard_samples):
            try:
                raw_prompt_arr = non_tensor_sample.get("raw_prompt", None)
                if raw_prompt_arr is None:
                    continue
                messages = list(deepcopy(raw_prompt_arr[0]))
                if not isinstance(messages, list):
                    continue
                sig = self._gepa_prompt_signature_from_messages(messages)
                signatures.add(sig)
            except Exception as exc:
                print(f"[GEPA] Warning: failed to compute signature for hard sample idx={sample_idx}: {exc}")
        return signatures

    def _finalize_epoch_prompt_optimization(self, epoch_idx: int):
        """Hook that runs at the end of each epoch."""
        hard_size = len(self._current_epoch_hard_samples)
        print(f"[GEPA] Epoch {epoch_idx} very_hard_dataset size = {hard_size}")

        if not self._gepa_enabled():
            self._prev_epoch_hard_signatures = set()
            self._prev_epoch_hard_signatures_train = set()
            self._prev_epoch_hard_signatures_dev = set()
            self._prev_epoch_hard_signatures_leftover = set()
            self._gepa_epoch_template = ""
            return

        if hard_size < self._gepa_cfg["hard_set_min_count"]:
            print(
                f"[GEPA] Skip optimization for epoch {epoch_idx}: "
                f"hard set size {hard_size} < hard_set_min_count (minimum required)."
            )
            self._prev_epoch_hard_signatures = set()
            self._prev_epoch_hard_signatures_train = set()
            self._prev_epoch_hard_signatures_dev = set()
            self._prev_epoch_hard_signatures_leftover = set()
            self._gepa_epoch_template = ""
            self._gepa_epoch_candidates = []
            self._gepa_epoch_dev_scores = []
            self._gepa_epoch_training_templates = []
            self._gepa_epoch_training_template_dev_scores = {}
            self._gepa_epoch_training_template_leftover_scores = {}
            return

        self._gepa_resume_info = {"epoch_idx": epoch_idx, "stage": "before_gepa_optimization"}
        self.global_steps -= 1
        self._save_checkpoint()
        print(
            f"[GEPA] Saved checkpoint before GEPA optimization in epoch {epoch_idx} "
            f"at global_step={self.global_steps}"
        )

        gepa_ran = self._run_gepa_prompt_optimization(epoch_idx=epoch_idx)

        if not gepa_ran:
            self._prev_epoch_hard_signatures = set()
            self._gepa_epoch_template = ""
            self._gepa_epoch_candidates = []
            self._gepa_epoch_dev_scores = []
            self._gepa_epoch_training_templates = []
            self._gepa_epoch_training_template_dev_scores = {}
            self._gepa_epoch_training_template_leftover_scores = {}
            return

        self._prev_epoch_hard_signatures = self._compute_prev_epoch_hard_signatures()

        self.global_steps += 1
        self._gepa_resume_info = {"epoch_idx": epoch_idx, "stage": "after_gepa_optimization"}
        self._save_checkpoint()
        print(
            f"[GEPA] Epoch {epoch_idx}: GEPA optimization finished; "
            f"saved GEPA checkpoint at global_step={self.global_steps}"
        )
        if (
            self.val_reward_fn is not None
            and self.config.trainer.test_freq > 0
            and (self.global_steps % self.config.trainer.test_freq == 0)
        ):
            val_metrics: dict = self._validate()
            self._logger.log(data=val_metrics, step=self.global_steps)
        self.global_steps += 1
        self._gepa_resume_info = None

    def _gepa_auto_budget(
        self,
        num_preds: int,
        num_candidates: int,
        valset_size: int,
        minibatch_size: int = 35,
        full_eval_steps: int = 5,
    ) -> int:
        """Mirror dspy.teleprompt.gepa.GEPA.auto_budget (without numpy).

        See dspy/teleprompt/gepa/gepa.py::GEPA.auto_budget.
        """
        import math

        num_trials = int(
            max(
                2 * (num_preds * 2) * math.log2(max(num_candidates, 1)),
                1.5 * num_candidates,
            )
        )
        if num_trials < 0 or valset_size < 0 or minibatch_size < 0:
            raise ValueError("num_trials, valset_size, and minibatch_size must be >= 0.")
        if full_eval_steps < 1:
            raise ValueError("full_eval_steps must be >= 1.")

        V = valset_size
        N = num_trials
        M = minibatch_size
        m = full_eval_steps

        total = V

        total += num_candidates * 5

        total += N * M
        if N == 0:
            return int(total)
        periodic_fulls = (N + 1) // (m) + 1
        extra_final = 1 if N < m else 0

        total += (periodic_fulls + extra_final) * V
        return int(total)

    def _compute_gepa_budget(self, train_size: int, dev_size: int) -> int:
        auto = self._gepa_cfg.get("auto")
        max_full = self._gepa_cfg.get("max_full_evals")
        max_calls = self._gepa_cfg.get("max_metric_calls")

        if auto is not None:
            num_candidates = self.auto_run_settings[auto]
            valset_size = dev_size if dev_size > 0 else train_size
            budget = self._gepa_auto_budget(
                num_preds=1,
                num_candidates=num_candidates,
                valset_size=valset_size,
            )
        elif max_full is not None:
            budget = max(1, int(max_full)) * max(1, train_size + dev_size)
        elif max_calls is not None:
            budget = max(1, int(max_calls))
        else:
            raise ValueError(
                "Exactly one of auto, max_full_evals, max_metric_calls should be set for GEPA."
            )
        return max(1, int(budget))

    def _run_gepa_prompt_optimization(self, epoch_idx: int) -> bool:
        """Run a GEPA-style optimization loop over the collected very hard samples.
        
        Returns:
            bool: True if GEPA optimization was actually performed, False otherwise.
        """
        hard_samples_raw = self._current_epoch_hard_samples[:]
        hard_size = len(hard_samples_raw)

        hard_samples: list[DataProto] = []
        for sample_idx, non_tensor_sample in enumerate(hard_samples_raw):
            try:
                hard_samples.append(DataProto.from_single_dict(non_tensor_sample))
            except Exception as exc:
                print(f"[GEPA] Warning: failed to rebuild hard sample idx={sample_idx}: {exc}")

        hard_size = len(hard_samples)

        if hard_size < self._gepa_cfg["hard_set_min_count"]:
            print(
                f"[GEPA] Skip optimization for epoch {epoch_idx}: "
                f"hard set size {hard_size} < hard_set_min_count (minimum required)."
            )
            return False

        train_ratio = self._gepa_cfg.get("train_ratio", self._gepa_cfg["hard_set_min_count"]//2)
        dev_ratio = self._gepa_cfg.get("dev_ratio", self._gepa_cfg["hard_set_min_count"]//2)

        train_samples, dev_samples, leftover_samples = self._split_gepa_dataset(
            hard_samples,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            total_size=hard_size,
        )

        train_signatures = self._compute_signatures_from_samples(train_samples)
        dev_signatures = self._compute_signatures_from_samples(dev_samples)
        leftover_signatures = self._compute_signatures_from_samples(leftover_samples)

        self._prev_epoch_hard_signatures_train = train_signatures
        self._prev_epoch_hard_signatures_dev = dev_signatures
        self._prev_epoch_hard_signatures_leftover = leftover_signatures

        print(
            f"[GEPA] Split hard set (size={hard_size}): "
            f"train={len(train_samples)} ({len(train_signatures)} unique sigs), "
            f"dev={len(dev_samples)} ({len(dev_signatures)} unique sigs), "
            f"leftover={len(leftover_samples)} ({len(leftover_signatures)} unique sigs)"
        )

        reflection_minibatch = self._gepa_cfg.get("reflection_minibatch_size", 3)
        rng = random.Random(epoch_idx + hard_size)

        candidates: list[str] = []
        parents: list[list[int | None]] = []
        val_subscores: list[list[float]] = []
        val_aggregate_scores: list[float] = []

        pareto_front_valset: list[float] = []
        program_at_pareto_front_valset: list[set[int]] = []

        def _init_pareto_state(initial_scores: list[float]):
            pareto_front_valset.clear()
            program_at_pareto_front_valset.clear()
            for s in initial_scores:
                pareto_front_valset.append(float(s))
                program_at_pareto_front_valset.append({0})

        def _update_pareto_with_new_candidate(new_idx: int, scores: list[float]):
            for task_idx, new_score in enumerate(scores):
                old_score = pareto_front_valset[task_idx]
                if new_score > old_score:
                    pareto_front_valset[task_idx] = float(new_score)
                    program_at_pareto_front_valset[task_idx] = {new_idx}
                elif new_score == old_score:
                    program_at_pareto_front_valset[task_idx].add(new_idx)

        def _select_pareto_candidate_idx(excluded: set[int] | None = None) -> int:
            """Mimic ParetoCandidateSelector: sample from union of Pareto programs with weights.
            
            Weights are based on how many samples each candidate is Pareto-optimal on.
            """
            excluded = excluded or set()
            candidate_counts: dict[int, int] = defaultdict(int)
            for s in program_at_pareto_front_valset:
                for candidate_idx in s:
                    if candidate_idx not in excluded:
                        candidate_counts[candidate_idx] += 1
            
            if not candidate_counts:
                available = [i for i in range(len(candidates)) if i not in excluded]
                if not available:
                    raise ValueError("No available candidates after exclusion")
                return rng.choice(available)
            
            candidate_list = list(candidate_counts.keys())
            weights = [candidate_counts[idx] for idx in candidate_list]
            return rng.choices(candidate_list, weights=weights, k=1)[0]

        timing_raw: dict[str, float] = {}

        def _full_eval_and_add(template: str, parent_idx: int | None) -> int:
            """Evaluate template on full dev set and add as a new candidate."""
            with marked_timer("gepa_full_eval", timing_raw, color="green"):
                records = self._evaluate_prompt_on_samples(dev_samples, template, rng)
            scores = [float(rec["reward"]) for rec in records]
            agg = float(np.mean(scores)) if scores else 0.0

            new_idx = len(candidates)
            candidates.append(template)
            parents.append([parent_idx])
            val_subscores.append(scores)
            val_aggregate_scores.append(agg)

            if new_idx == 0:
                _init_pareto_state(scores)
            else:
                if scores:
                    _update_pareto_with_new_candidate(new_idx, scores)

            return new_idx

        def _sum_rewards(records: list[dict]) -> float:
            return float(sum(rec.get("reward", 0.0) for rec in records)) if records else float("-inf")

        seed_template = ""
        seed_idx = _full_eval_and_add(seed_template, parent_idx=None)
        assert seed_idx == 0

        used_train_indices: set[int] = set()
        gepa_select_pareto_k = max(1, int(self._gepa_cfg.get("gepa_select_pareto_k", 1)))

        budget = self._compute_gepa_budget(len(train_samples), len(dev_samples))
        remaining_budget = budget
        iteration = 0
        
        print(
            f"[GEPA] Starting gepa prompt evolution with budget={budget}, "
            f"reflection_minibatch={reflection_minibatch}, dev_samples={len(dev_samples)}"
        )
        
        if self._logger is not None:
            gepa_init_metrics = {
                f"gepa/epoch_{epoch_idx}/init/budget": budget,
                f"gepa/epoch_{epoch_idx}/init/train_samples": len(train_samples),
                f"gepa/epoch_{epoch_idx}/init/dev_samples": len(dev_samples),
                f"gepa/epoch_{epoch_idx}/init/reflection_minibatch": reflection_minibatch,
            }
            self._logger.log(data=gepa_init_metrics, step=self.global_steps)

        self._reflection_llm = None
        if self._gepa_cfg.get("use_reference_model_to_reflect", False):
            print("[GEPA] Creating vLLM LLM instance for reflection model...")
            model_path = self.config.actor_rollout_ref.model.path
            tensor_parallel_size = self.config.trainer.n_gpus_per_node
            max_model_len = self._gepa_cfg.get("reflection_max_model_length", None)
            gpu_memory_utilization = self._gepa_cfg.get("reflection_gpu_memory_utilization", 0.2)
            
            self._reflection_llm = self._create_vllm_llm(
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
            print(f"[GEPA] vLLM LLM instance created successfully")

        while True:
            if len(train_samples) == 0:
                print("[GEPA] No train samples available for reflection minibatch; stopping early.")
                break

            selected_prog_ids = []
            selected_prog_ids_set = set()
            
            all_available = set()
            for s in program_at_pareto_front_valset:
                all_available.update(s)
            if not all_available:
                all_available = set(range(len(candidates)))
            
            if len(all_available) <= gepa_select_pareto_k:
                selected_prog_ids = list(all_available)
            else:
                for _ in range(gepa_select_pareto_k):
                    if len(candidates) == 0:
                        break
                    try:
                        selected_prog_id = _select_pareto_candidate_idx(excluded=selected_prog_ids_set)
                        selected_prog_ids.append(selected_prog_id)
                        selected_prog_ids_set.add(selected_prog_id)
                    except ValueError:
                        break
            
            if len(selected_prog_ids) == 0:
                print("[GEPA] No candidates available; stopping early.")
                break

            is_last_iteration = False
            if iteration >= 2:
                estimated_cost = reflection_minibatch * len(selected_prog_ids) + len(dev_samples) * len(selected_prog_ids)
                
                if remaining_budget < estimated_cost:
                    if remaining_budget > 0:
                        is_last_iteration = True
                        print(
                            f"[GEPA][Epoch {epoch_idx}] Iteration {iteration + 1}: "
                            f"Budget insufficient (remaining={remaining_budget}, estimated={estimated_cost}), "
                            f"but proceeding with final iteration"
                        )
                    else:
                        print(
                            f"[GEPA][Epoch {epoch_idx}] Budget exhausted (remaining={remaining_budget}). "
                            f"Stopping after {iteration} iterations."
                        )
                        break

            candidate_data = []
            for prog_id in selected_prog_ids:
                current_template = candidates[prog_id]
                mb_size = min(reflection_minibatch, len(train_samples))
                available_indices = [i for i in range(len(train_samples)) if i not in used_train_indices]
                if len(available_indices) < mb_size:
                    available_indices = list(range(len(train_samples)))
                mb_indices = rng.sample(available_indices, k=min(mb_size, len(available_indices)))
                used_train_indices.update(mb_indices)
                minibatch = [train_samples[i] for i in mb_indices]
                candidate_data.append((prog_id, current_template, minibatch, mb_indices))

            template_minibatch_pairs = [(template, minibatch) for _, template, minibatch, _ in candidate_data]
            minibatch_rewards_before = []
            with marked_timer("gepa_mb_before", timing_raw, color="yellow"):
                all_reward_records = self._evaluate_multiple_prompts_on_samples_batch(
                    template_minibatch_pairs, rng
                )
                for idx, (prog_id, current_template, minibatch, _) in enumerate(candidate_data):
                    minibatch_reward_info_before = all_reward_records[idx]
                    self._attach_inputs_to_records(minibatch, minibatch_reward_info_before)
                    score_before = _sum_rewards(minibatch_reward_info_before)
                    minibatch_rewards_before.append((prog_id, current_template, minibatch, minibatch_reward_info_before, score_before))
                    print(
                        f"[GEPA][Epoch {epoch_idx}] Iteration {iteration + 1} "
                        f"Template {prog_id} minibatch eval BEFORE: score={score_before:.4f}, "
                        f"current_template={current_template!r}"
                    )

            improved_templates = []
            with marked_timer("gepa_reflect", timing_raw, color="purple"):
                reflection_prompts = []
                for prog_id, current_template, minibatch, minibatch_reward_info_before, score_before in minibatch_rewards_before:
                    dataset_examples = [
                        self._record_to_instruction_example(rec) for rec in minibatch_reward_info_before
                    ]
                    reflection_prompt = self._build_reflection_prompt(
                        template=current_template,
                        dataset_examples=dataset_examples,
                    )
                    print("\n\n" + "=" * 100 + f"\n[GEPA] reflection_prompt:\n{reflection_prompt}\n" + "=" * 100 + "\n\n")
                    reflection_prompts.append(reflection_prompt)
                
                improved_template_list = self._sample_instructions_from_prompts_batch(reflection_prompts)
                
                for idx, (prog_id, current_template, minibatch, _, score_before) in enumerate(minibatch_rewards_before):
                    improved_template = improved_template_list[idx].strip() if idx < len(improved_template_list) else ""
                    if not improved_template:
                        improved_template = current_template
                    improved_templates.append((prog_id, current_template, improved_template, minibatch, score_before))
                    print(
                        f"[GEPA][Epoch {epoch_idx}] Iteration {iteration + 1} "
                        f"Template {prog_id} improved_template={improved_template!r}"
                    )

            accepted_candidates = []
            with marked_timer("gepa_mb_after", timing_raw, color="yellow"):
                improved_template_minibatch_pairs = [
                    (improved_template, minibatch) 
                    for _, _, improved_template, minibatch, _ in improved_templates
                ]
                all_reward_records_after = self._evaluate_multiple_prompts_on_samples_batch(
                    improved_template_minibatch_pairs, rng
                )
                
                for idx, (prog_id, current_template, improved_template, minibatch, score_before) in enumerate(improved_templates):
                    minibatch_reward_info_after = all_reward_records_after[idx]
                    score_after = _sum_rewards(minibatch_reward_info_after)
                    print(
                        f"[GEPA][Epoch {epoch_idx}] Iteration {iteration + 1} "
                        f"Template {prog_id} minibatch eval AFTER: score={score_after:.4f}, "
                        f"improved_template={improved_template!r}"
                    )
                    
                    if self._logger is not None:
                        gepa_metrics = {
                            f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/minibatch_score_before": score_before,
                            f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/minibatch_score_after": score_after,
                            f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/minibatch_score_improvement": score_after - score_before,
                            f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/current_template": current_template,
                            f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/improved_template": improved_template,
                        }
                        self._logger.log(data=gepa_metrics, step=self.global_steps)

                    if score_after > score_before:
                        accepted_candidates.append((prog_id, improved_template, score_before, score_after))
                    else:
                        print(
                            f"[GEPA][Epoch {epoch_idx}] Iteration {iteration + 1} "
                            f"Template {prog_id} rejected candidate (minibatch score {score_after:.4f} <= {score_before:.4f})."
                        )
                        if self._logger is not None:
                            gepa_metrics = {
                                f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/candidate_status": "rejected",
                                f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/minibatch_score_before": score_before,
                                f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/minibatch_score_after": score_after,
                            }
                            self._logger.log(data=gepa_metrics, step=self.global_steps)

            if accepted_candidates:
                with marked_timer("gepa_full_eval", timing_raw, color="green"):
                    template_dev_pairs = [
                        (improved_template, dev_samples)
                        for _, improved_template, _, _ in accepted_candidates
                    ]
                    all_records_list = self._evaluate_multiple_prompts_on_samples_batch(
                        template_dev_pairs, rng
                    )
                
                for (prog_id, improved_template, score_before, score_after), records in zip(
                    accepted_candidates, all_records_list
                ):
                    scores = [float(rec["reward"]) for rec in records]
                    agg = float(np.mean(scores)) if scores else 0.0

                    new_idx = len(candidates)
                    candidates.append(improved_template)
                    parents.append([prog_id])
                    val_subscores.append(scores)
                    val_aggregate_scores.append(agg)

                    if new_idx == 0:
                        _init_pareto_state(scores)
                    else:
                        if scores:
                            _update_pareto_with_new_candidate(new_idx, scores)

                    best_idx = max(range(len(val_aggregate_scores)), key=lambda i: val_aggregate_scores[i])
                    best_score = val_aggregate_scores[best_idx]

                    print(
                        f"[GEPA][Epoch {epoch_idx}] Iteration {iteration + 1} "
                        f"Template {prog_id} accepted candidate idx={new_idx}; dev_score={agg:.4f} best_score={best_score:.4f}"
                    )
                    
                    if self._logger is not None:
                        gepa_metrics = {
                            f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/candidate_status": "accepted",
                            f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/candidate_idx": new_idx,
                            f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/dev_score": agg,
                            f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/best_score": best_score,
                            f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/best_idx": best_idx,
                            f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/template_{prog_id}/num_candidates": len(candidates),
                            f"gepa/epoch_{epoch_idx}/dev_score": agg,
                            f"gepa/epoch_{epoch_idx}/best_score": best_score,
                            f"gepa/epoch_{epoch_idx}/num_candidates": len(candidates),
                        }
                        self._logger.log(data=gepa_metrics, step=self.global_steps)

            if iteration >= 2:
                cost_reflection_minibatch = reflection_minibatch * len(selected_prog_ids)
                cost_dev_eval = len(accepted_candidates) * len(dev_samples)
                iteration_cost = cost_reflection_minibatch + cost_dev_eval
                
                remaining_budget -= iteration_cost
                
                print(
                    f"[GEPA][Epoch {epoch_idx}] Iteration {iteration + 1} completed: "
                    f"cost={iteration_cost} (reflection={cost_reflection_minibatch}, dev={cost_dev_eval}), "
                    f"remaining_budget={remaining_budget}"
                )
                
                if self._logger is not None:
                    gepa_metrics = {
                        f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/cost": iteration_cost,
                        f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/cost_reflection_minibatch": cost_reflection_minibatch,
                        f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/cost_dev_eval": cost_dev_eval,
                        f"gepa/epoch_{epoch_idx}/iteration_{iteration + 1}/remaining_budget": remaining_budget,
                    }
                    self._logger.log(data=gepa_metrics, step=self.global_steps)
                
                if remaining_budget <= 0 and not is_last_iteration:
                    print(
                        f"[GEPA][Epoch {epoch_idx}] Budget exhausted. "
                        f"Stopping after {iteration + 1} iterations."
                    )
                    break
            
            iteration += 1

        if hasattr(self, "_reflection_llm") and self._reflection_llm is not None:
            print("[GEPA] Deleting vLLM LLM instance and freeing GPU memory...")
            try:
                if hasattr(self._reflection_llm, "llm_engine") and hasattr(self._reflection_llm.llm_engine, "shutdown"):
                    try:
                        self._reflection_llm.llm_engine.shutdown()
                    except Exception:
                        pass
                
                del self._reflection_llm
                self._reflection_llm = None
                print("[GEPA] vLLM LLM instance deleted and GPU memory freed successfully.")
            except Exception as e:
                print(f"[GEPA] Warning: Failed to delete vLLM LLM instance gracefully: {e}")
                try:
                    self._reflection_llm = None
                except Exception:
                    pass

        if val_aggregate_scores:
            sorted_indices = sorted(
                range(len(val_aggregate_scores)),
                key=lambda i: val_aggregate_scores[i],
                reverse=True,
            )
            best_idx = sorted_indices[0]
            best_template = candidates[best_idx]
            best_score = val_aggregate_scores[best_idx]
        else:
            return False

        self._gepa_epoch_candidates = list(candidates)
        self._gepa_epoch_dev_scores = list(val_aggregate_scores)

        self._gepa_epoch_template = best_template
        print(f"[GEPA] Epoch {epoch_idx} chosen template (score={best_score:.4f}): {best_template!r}")
        
        if self._logger is not None:
            gepa_final_metrics = {
                f"gepa/epoch_{epoch_idx}/final/best_template": best_template,
                f"gepa/epoch_{epoch_idx}/final/best_score": best_score,
                f"gepa/epoch_{epoch_idx}/final/best_idx": best_idx,
                f"gepa/epoch_{epoch_idx}/final/total_candidates": len(candidates),
                f"gepa/epoch_{epoch_idx}/final/total_iterations": iteration,
            }
            if timing_raw:
                for timing_key, timing_value in timing_raw.items():
                    gepa_final_metrics[f"gepa/epoch_{epoch_idx}/final/timing/{timing_key}"] = timing_value
            self._logger.log(data=gepa_final_metrics, step=self.global_steps)

        unused_train_samples = [
            train_samples[i] for i in range(len(train_samples)) if i not in used_train_indices
        ]
        grpo_samples = leftover_samples + unused_train_samples
        if len(grpo_samples) == 0:
            grpo_samples = hard_samples

        if self._gepa_cfg.get("select_top_k_template_to_train", False):
            k = int(self._gepa_cfg.get("best_template_k", 3))
            k = max(1, min(k, len(sorted_indices)))
            top_k_indices = sorted_indices[:k]
            training_templates = [candidates[i] for i in top_k_indices]
            print(
                f"[GEPA] Using top-{k} templates for GRPO training after epoch {epoch_idx}."
            )
        else:
            training_templates = [best_template]

        training_template_dev_scores: dict[str, float] = {}
        if self._gepa_epoch_dev_scores:
            if self._gepa_cfg.get("select_top_k_template_to_train", False):
                for idx in top_k_indices:
                    template = candidates[idx]
                    training_template_dev_scores[template] = self._gepa_epoch_dev_scores[idx]
            else:
                training_template_dev_scores[best_template] = best_score

        self._gepa_epoch_training_templates = list(training_templates)
        self._gepa_epoch_training_template_dev_scores = training_template_dev_scores

        depth_cache = {}
        def _compute_depth(idx: int) -> int:
            """Compute depth of a node (distance from root, root has depth 1)."""
            if idx < 0 or idx >= len(parents):
                return 0
            if idx in depth_cache:
                return depth_cache[idx]
            if idx == 0 or not parents[idx] or parents[idx][0] is None:
                depth_cache[idx] = 1
                return 1
            parent_idx = parents[idx][0]
            depth = _compute_depth(parent_idx) + 1
            depth_cache[idx] = depth
            return depth
        
        max_tree_depth = 0
        for i in range(len(candidates)):
            depth = _compute_depth(i)
            max_tree_depth = max(max_tree_depth, depth)
        
        print(f"[GEPA] Total candidates (templates) generated: {len(candidates)}")
        print(f"[GEPA] Total iterations: {iteration}")
        print(f"[GEPA] Tree depth (max depth from root): {max_tree_depth}")
        
        if len(val_aggregate_scores) > 0:
            init_template_score = val_aggregate_scores[0]
            init_template = candidates[0] if len(candidates) > 0 else ""
            print(f"[GEPA] Initial template (dev score): {init_template_score:.4f} | template={init_template!r}")
        
        print(f"[GEPA] Best template (dev score): {best_score:.4f} | template={best_template!r}")
        
        if self._gepa_cfg.get("select_top_k_template_to_train", False):
            print(f"[GEPA] Top-{k} templates (dev scores):")
            for rank, idx in enumerate(top_k_indices, 1):
                template = candidates[idx]
                score = val_aggregate_scores[idx]
                print(f"[GEPA]   Rank {rank}: score={score:.4f} | template={template!r}")

        training_template_leftover_scores: dict[str, float] = {}
        if self._gepa_cfg.get("evaluate_pass_at_k_before_next_epoch", True):
            k_for_pass = self.config.actor_rollout_ref.rollout.n
            print(f"[GEPA] Evaluating pass@{k_for_pass} on grpo_samples (size={len(grpo_samples)})")
            
            initial_template = ""
            print(f"[GEPA] Evaluating initial template (empty string) pass@{k_for_pass}...")
            initial_pass_at_k = self._compute_pass_at_k_with_rollout(
                grpo_samples, initial_template, k_for_pass, rng
            )
            print(f"[GEPA] Initial template (\"\") pass@{k_for_pass} = {initial_pass_at_k:.4f}")
            
            if self._logger is not None:
                self._logger.log(
                    data={
                        f"gepa/epoch_{epoch_idx}/pass_at_k/initial_template": initial_pass_at_k,
                        f"gepa/epoch_{epoch_idx}/pass_at_k/k": k_for_pass,
                    },
                    step=self.global_steps,
                )
            
            for template_idx, template in enumerate(training_templates):
                print(f"[GEPA] Evaluating training template {template_idx + 1}/{len(training_templates)} pass@{k_for_pass}...")
                template_pass_at_k = self._compute_pass_at_k_with_rollout(
                    grpo_samples, template, k_for_pass, rng
                )
                training_template_leftover_scores[template] = template_pass_at_k
                template_name = f"template_{template_idx + 1}" if len(training_templates) > 1 else "best_template"
                print(f"[GEPA] Training {template_name} pass@{k_for_pass} = {template_pass_at_k:.4f}")
                
                if self._logger is not None:
                    self._logger.log(
                        data={
                            f"gepa/epoch_{epoch_idx}/pass_at_k/{template_name}": template_pass_at_k,
                            f"gepa/epoch_{epoch_idx}/pass_at_k/{template_name}_template": template,
                        },
                        step=self.global_steps,
                    )
        else:
            print(f"[GEPA] Skipping pass@k evaluation before training (evaluate_pass_at_k_before_next_epoch=False)")

        self._gepa_epoch_training_template_leftover_scores = training_template_leftover_scores

        return True

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[:generations_to_log]

        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
        
        if isinstance(self.val_reward_fn, GroupedValidRewardManager):
            answer_scores = self.val_reward_fn.get_grouped_scores()
            scores = [s1 + s2 for s1, s2 in zip(answer_scores, reward_extra_infos_dict["format_score"])]
            sample_scores = scores
            reward_extra_infos_dict["reward"] = scores[:]
            reward_extra_infos_dict["score"] = scores[:]
            reward_extra_infos_dict["answer_score"] = answer_scores[:]

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        gepa_state = {
            "gepa_epoch_template": self._gepa_epoch_template,
            "gepa_epoch_candidates": getattr(self, "_gepa_epoch_candidates", []),
            "gepa_epoch_dev_scores": getattr(self, "_gepa_epoch_dev_scores", []),
            "gepa_epoch_training_templates": getattr(self, "_gepa_epoch_training_templates", []),
            "gepa_epoch_training_template_dev_scores": getattr(
                self, "_gepa_epoch_training_template_dev_scores", {}
            ),
            "gepa_epoch_training_template_leftover_scores": getattr(
                self, "_gepa_epoch_training_template_leftover_scores", {}
            ),
            "current_epoch_hard_samples": self._current_epoch_hard_samples,
            "gepa_epoch_uid_cache": list(self._gepa_epoch_uid_cache),
            "prev_epoch_hard_signatures": list(getattr(self, "_prev_epoch_hard_signatures", set())),
            "prev_epoch_hard_signatures_train": list(getattr(self, "_prev_epoch_hard_signatures_train", set())),
            "prev_epoch_hard_signatures_dev": list(getattr(self, "_prev_epoch_hard_signatures_dev", set())),
            "prev_epoch_hard_signatures_leftover": list(getattr(self, "_prev_epoch_hard_signatures_leftover", set())),
            "gepa_resume_info": getattr(self, "_gepa_resume_info", None),
        }
        gepa_state_path = os.path.join(local_global_step_folder, "gepa_state.pt")
        torch.save(gepa_state, gepa_state_path)

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))
        self._last_checkpoint_step = self.global_steps

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        gepa_state_path = os.path.join(global_step_folder, "gepa_state.pt")
        if os.path.exists(gepa_state_path):
            gepa_state = torch.load(gepa_state_path, weights_only=False)
            self._gepa_epoch_template = gepa_state.get("gepa_epoch_template", "")
            self._current_epoch_hard_samples = gepa_state.get("current_epoch_hard_samples", [])
            uid_cache_list = gepa_state.get("gepa_epoch_uid_cache", [])
            self._gepa_epoch_uid_cache = set(uid_cache_list) if uid_cache_list else set()

            self._gepa_epoch_candidates = gepa_state.get("gepa_epoch_candidates", [])
            self._gepa_epoch_dev_scores = gepa_state.get("gepa_epoch_dev_scores", [])
            self._gepa_epoch_training_templates = gepa_state.get("gepa_epoch_training_templates", [])
            self._gepa_epoch_training_template_dev_scores = gepa_state.get(
                "gepa_epoch_training_template_dev_scores", {}
            )
            self._gepa_epoch_training_template_leftover_scores = gepa_state.get(
                "gepa_epoch_training_template_leftover_scores", {}
            )

            prev_sigs = gepa_state.get("prev_epoch_hard_signatures", [])
            self._prev_epoch_hard_signatures = set(prev_sigs) if prev_sigs else set()

            self._gepa_resume_info = gepa_state.get("gepa_resume_info")

            self.is_gepa_loading_from_checkpoint = True
            print(
                f"Loaded GEPA state: template={self._gepa_epoch_template}, "
                f"hard samples={len(self._current_epoch_hard_samples)}, "
                f"uid cache size={len(self._gepa_epoch_uid_cache)}, "
                f"prev hard signatures={len(self._prev_epoch_hard_signatures)}"
            )
        else:
            print(f"Warning: No GEPA state found at {gepa_state_path}, will start from scratch")
            self._gepa_epoch_template = ""
            self._current_epoch_hard_samples = []
            self._gepa_epoch_uid_cache = set()
            self._prev_epoch_hard_signatures = set()
            self._gepa_epoch_candidates = []
            self._gepa_epoch_dev_scores = []
            self._gepa_epoch_training_templates = []
            self._gepa_epoch_training_template_dev_scores = {}
            self._gepa_epoch_training_template_leftover_scores = {}
            self.is_gepa_loading_from_checkpoint = False
            self._gepa_resume_info = None

        self._last_checkpoint_step = self.global_steps

    def _resume_pending_gepa_if_needed(self):
        """If a checkpoint occurred mid-GEPA finalization, finish the remaining steps."""
        if not self._gepa_resume_info:
            return

        epoch_idx = self._gepa_resume_info.get("epoch_idx")
        stage = self._gepa_resume_info.get("stage")
        if stage != "before_gepa_optimization":
            return

        print(f"[GEPA] Resuming pending GEPA finalization for epoch {epoch_idx} at step {self.global_steps}")
        gepa_ran = self._run_gepa_prompt_optimization(epoch_idx=epoch_idx)

        if not gepa_ran:
            self._prev_epoch_hard_signatures = set()
            self._prev_epoch_hard_signatures_train = set()
            self._prev_epoch_hard_signatures_dev = set()
            self._prev_epoch_hard_signatures_leftover = set()
            self._gepa_epoch_template = ""
            self._gepa_epoch_candidates = []
            self._gepa_epoch_dev_scores = []
            self._gepa_epoch_training_templates = []
            self._gepa_epoch_training_template_dev_scores = {}
            self._gepa_epoch_training_template_leftover_scores = {}
            print(
                f"[GEPA] Epoch {epoch_idx}: GEPA optimization did not run successfully "
                "(resume path); clearing GEPA state."
            )
        else:
            self._prev_epoch_hard_signatures = self._compute_prev_epoch_hard_signatures()
            self.global_steps += 1

        self._gepa_resume_info = {"epoch_idx": epoch_idx, "stage": "after_gepa_optimization"}
        self._save_checkpoint()
        print(f"[GEPA] Resume path completed and checkpointed for epoch {epoch_idx} step {self.global_steps}")
        if (
            self.val_reward_fn is not None
            and self.config.trainer.test_freq > 0
            and (self.global_steps % self.config.trainer.test_freq == 0)
        ):
            val_metrics: dict = self._validate()
            self._logger.log(data=val_metrics, step=self.global_steps)

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self._logger = logger

        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise RuntimeError(
                "This trainer variant only supports algorithm.adv_estimator == AdvantageEstimator.GRPO when GEPA logic is enabled."
            )
        if not self.config.data.get("return_raw_chat", False):
            raise RuntimeError(
                "This trainer variant only supports data.return_raw_chat == True when GEPA logic is enabled."
            )
        entropy_coeff = self.config.actor_rollout_ref.actor.get("entropy_coeff", 0.)
        if entropy_coeff is not None and entropy_coeff != 0.:
            raise RuntimeError(
                "actor_rollout_ref.actor.entropy_coeff must be zero!"
            )

        self.global_steps = 0

        self._load_checkpoint()
        self._resume_pending_gepa_if_needed()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            if self.is_gepa_loading_from_checkpoint and self._gepa_resume_info is not None:
                stage = self._gepa_resume_info.get("stage")
                if stage == "before_gepa_optimization":
                    raise RuntimeError(f"Not run _resume_pending_gepa_if_needed, Global_Step={self.global_steps}")
                elif stage == "after_gepa_optimization":
                    self._reset_epoch_gepa_state()
                else:
                    raise RuntimeError(f"Unknown stage: {stage}")
            elif self.is_gepa_loading_from_checkpoint and self._gepa_resume_info is None:
                print(f"Keeping GEPA state for epoch {epoch}, because checkpoint was saved during training (not at epoch end).")
            else:
                self._reset_epoch_gepa_state()
            self.is_gepa_loading_from_checkpoint = False
            self._gepa_resume_info = None
            for batch_dict in self.train_dataloader:
                print(f"Epoch: {epoch}   Global Step: {self.global_steps}")
                metrics = {}
                timing_raw = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                base_hard_mask = None
                if self._gepa_enabled() and self._prev_epoch_hard_signatures:
                    assert batch.non_tensor_batch is not None and "raw_prompt" in batch.non_tensor_batch
                    raw_prompt_arr = batch.non_tensor_batch["raw_prompt"]
                    if isinstance(raw_prompt_arr, np.ndarray):
                        bsz = raw_prompt_arr.shape[0]
                        mask = np.zeros(bsz, dtype=bool)
                        for i in range(bsz):
                            try:
                                messages = list(deepcopy(raw_prompt_arr[i]))
                                sig = self._gepa_prompt_signature_from_messages(messages)
                                if sig in self._prev_epoch_hard_signatures:
                                    mask[i] = True
                            except Exception as exc:
                                print(f"[GEPA] Warning: failed to build signature for batch sample idx={i}: {exc}")
                        if mask.any():
                            base_hard_mask = mask

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                gen_batch.meta_info["global_steps"] = self.global_steps
                n_rollouts = self.config.actor_rollout_ref.rollout.n
                gen_batch = gen_batch.repeat(repeat_times=n_rollouts, interleave=True)

                gen_batch.non_tensor_batch["__gepa_template_applied__"] = np.zeros(len(gen_batch), dtype=bool)

                if self._gepa_enabled() and base_hard_mask is not None and n_rollouts > 0:
                    templates = []
                    if self._gepa_cfg.get("select_top_k_template_to_train", False) and getattr(self, "_gepa_epoch_training_templates", None):
                        templates = list(self._gepa_epoch_training_templates)
                    else:
                        if getattr(self, "_gepa_epoch_training_templates", None):
                            templates = list(self._gepa_epoch_training_templates)
                        elif self._gepa_epoch_template:
                            templates = [self._gepa_epoch_template]

                    if len(templates) > 0:
                        base_batch_size = len(batch)

                        if "raw_prompt" in gen_batch.non_tensor_batch:
                            original_raw_prompts = gen_batch.non_tensor_batch["raw_prompt"]
                            gen_batch.non_tensor_batch[NO_TEMPLATE_RAW_PROMPT_KEY] = original_raw_prompts.copy()

                        if len(templates) == 1 or n_rollouts <= 1:
                            template_indices_per_rollout = [0] * max(1, n_rollouts)
                        else:
                            if self._gepa_cfg.get("left_1_no_template_rollout", False):
                                effective_rollouts = n_rollouts - 1
                                k = len(templates)
                                base = effective_rollouts // k
                                rem = effective_rollouts % k
                                counts = [base + (1 if i < rem else 0) for i in range(k)]
                                template_indices_per_rollout = [-1] + [
                                    idx for idx, c in enumerate(counts) for _ in range(c)
                                ]
                                template_indices_per_rollout = template_indices_per_rollout[:n_rollouts]
                            else:
                                k = len(templates)
                                base = n_rollouts // k
                                rem = n_rollouts % k
                                counts = [base + (1 if i < rem else 0) for i in range(k)]
                                template_indices_per_rollout = [
                                    idx for idx, c in enumerate(counts) for _ in range(c)
                                ]
                                template_indices_per_rollout = template_indices_per_rollout[:n_rollouts]

                        template_applied_mask = gen_batch.non_tensor_batch["__gepa_template_applied__"]
                        total_rows = len(gen_batch)
                        for row_idx in range(total_rows):
                            base_idx = row_idx // n_rollouts
                            if not base_hard_mask[base_idx]:
                                continue
                            rollout_idx = row_idx % n_rollouts
                            if rollout_idx >= len(template_indices_per_rollout):
                                continue
                            tpl_idx = template_indices_per_rollout[rollout_idx]

                            if tpl_idx == -1:
                                continue

                            tpl = templates[tpl_idx]

                            if tpl.strip():
                                template_applied_mask[row_idx] = True

                            updated_single = self._apply_prompt_template_to_sample(
                                gen_batch[row_idx:row_idx+1],
                                prompt_template=tpl,
                                truncation="left",
                            )

                            for key_name in ["input_ids", "attention_mask", "position_ids"]:
                                if key_name in updated_single.batch and key_name in gen_batch.batch:
                                    gen_batch.batch[key_name][row_idx] = updated_single.batch[key_name][0]

                            for key_name in ["raw_prompt_ids", "raw_prompt"]:
                                if (
                                    key_name in updated_single.non_tensor_batch
                                    and key_name in gen_batch.non_tensor_batch
                                ):
                                    gen_batch.non_tensor_batch[key_name][row_idx] = updated_single.non_tensor_batch[key_name][0]
                    else:
                        raise RuntimeError("len(templates) is zero")

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    uid_to_sample_map: dict[str, DataProto] = {}
                    base_batch_size = len(batch.batch)
                    if self._gepa_enabled():
                        uid_list = []
                        for sample_idx in range(base_batch_size):
                            uid = str(uuid.uuid4())
                            uid_list.append(uid)
                            uid_to_sample_map[uid] = self._extract_sample_from_batch_dict(batch_dict, sample_idx)
                    else:
                        uid_list = [str(uuid.uuid4()) for _ in range(base_batch_size)]
                    batch.non_tensor_batch["uid"] = np.array(uid_list, dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    if self._gepa_enabled() and base_hard_mask is not None and self._gepa_cfg.get("enable_template_ratio_mask", False):
                        with marked_timer("template_ratio", timing_raw, color="magenta"):
                            n_rollouts = self.config.actor_rollout_ref.rollout.n
                            base_batch_size = len(base_hard_mask)
                            total_batch_size = len(batch)

                            if "__gepa_template_applied__" not in batch.non_tensor_batch:
                                print(f"[GEPA] Warning: __gepa_template_applied__ not found, skipping template_ratio_mask calculation")
                            else:
                                template_applied_mask = batch.non_tensor_batch["__gepa_template_applied__"]

                                template_applied_indices = np.where(template_applied_mask)[0]
                                num_template_applied = len(template_applied_indices)

                                if num_template_applied > 0:
                                    template_applied_batch = batch[template_applied_indices.tolist()]

                                    world_size = self.actor_rollout_wg.world_size
                                    template_applied_batch_padded, pad_size = pad_dataproto_to_divisor(
                                        template_applied_batch, world_size
                                    )

                                    with_template_log_prob_data_padded = self.actor_rollout_wg.compute_log_prob(template_applied_batch_padded)
                                    with_template_log_prob_data = unpad_dataproto(with_template_log_prob_data_padded, pad_size)
                                    with_template_log_prob = with_template_log_prob_data.batch["old_log_probs"]

                                    batch = self._rebuild_batch_input(batch)

                                    no_template_batch = batch[template_applied_indices.tolist()]

                                    no_template_batch_padded, pad_size = pad_dataproto_to_divisor(
                                        no_template_batch, world_size
                                    )

                                    no_template_log_prob_data_padded = self.actor_rollout_wg.compute_log_prob(no_template_batch_padded)
                                    no_template_log_prob_data = unpad_dataproto(no_template_log_prob_data_padded, pad_size)
                                    no_template_log_prob = no_template_log_prob_data.batch["old_log_probs"]

                                    template_applied_ratio = torch.exp(no_template_log_prob - with_template_log_prob)

                                    responses = batch.batch["responses"]
                                    response_length = responses.size(1)
                                    template_ratio = torch.ones(total_batch_size, response_length, device=template_applied_ratio.device, dtype=template_applied_ratio.dtype)
                                    template_ratio[template_applied_indices, :] = template_applied_ratio

                                    cliprange_low = self._gepa_cfg.get("template_ratio_cliprange_low", 0.2)
                                    cliprange_high = self._gepa_cfg.get("template_ratio_cliprange_high", 0.2)
                                    soft_clip = self._gepa_cfg.get("soft_clip", False)
                                    lower_bound = 1.0 - cliprange_low
                                    upper_bound = 1.0 + cliprange_high

                                    template_ratio_mask = torch.ones_like(template_ratio)

                                    if soft_clip:
                                        upper_exceed_mask = template_ratio > upper_bound
                                        template_ratio_mask[upper_exceed_mask] = upper_bound / template_ratio[upper_exceed_mask]

                                        lower_exceed_mask = template_ratio < lower_bound
                                        template_ratio_mask[lower_exceed_mask] = lower_bound / template_ratio[lower_exceed_mask]
                                    else:
                                        out_of_range_mask = (template_ratio < lower_bound) | (template_ratio > upper_bound)
                                        template_ratio_mask[out_of_range_mask] = 0.0

                                    response_mask = batch.batch["response_mask"]

                                    template_applied_response_mask = response_mask[template_applied_indices, :]
                                    template_applied_ratio_for_stats = template_applied_ratio * template_applied_response_mask
                                    template_applied_mask_for_stats = template_ratio_mask[template_applied_indices, :] * template_applied_response_mask

                                    template_applied_valid_token_count = template_applied_response_mask.sum().item()
                                    total_valid_token_count = response_mask.sum().item()

                                    if template_applied_valid_token_count > 0:
                                        valid_ratios = template_applied_ratio_for_stats[template_applied_response_mask.bool()]
                                        template_ratio_max = valid_ratios.max().item()
                                        template_ratio_min = valid_ratios.min().item()
                                        template_ratio_mean = valid_ratios.mean().item()
                                        template_ratio_std = valid_ratios.std().item()

                                        clipped_mask_in_template_applied = (template_applied_mask_for_stats != template_applied_response_mask)
                                        clipped_token_count_in_template_applied = clipped_mask_in_template_applied.sum().item()
                                        clipped_ratio_in_template_applied = clipped_token_count_in_template_applied / template_applied_valid_token_count

                                        clipped_mask_in_total = ((template_ratio_mask * response_mask) != response_mask)
                                        clipped_token_count_in_total = clipped_mask_in_total.sum().item()
                                        clipped_ratio_in_total = clipped_token_count_in_total / total_valid_token_count if total_valid_token_count > 0 else 0.0

                                        lower_clipped_mask_in_template_applied = (template_applied_ratio < lower_bound) & template_applied_response_mask.bool()
                                        upper_clipped_mask_in_template_applied = (template_applied_ratio > upper_bound) & template_applied_response_mask.bool()
                                        lower_clipped_count_in_template_applied = lower_clipped_mask_in_template_applied.sum().item()
                                        upper_clipped_count_in_template_applied = upper_clipped_mask_in_template_applied.sum().item()
                                        lower_clipped_token_ratio_in_template_applied = lower_clipped_count_in_template_applied / template_applied_valid_token_count
                                        upper_clipped_token_ratio_in_template_applied = upper_clipped_count_in_template_applied / template_applied_valid_token_count

                                        lower_clipped_mask_in_total = (template_ratio < lower_bound) & response_mask.bool()
                                        upper_clipped_mask_in_total = (template_ratio > upper_bound) & response_mask.bool()
                                        lower_clipped_count_in_total = lower_clipped_mask_in_total.sum().item()
                                        upper_clipped_count_in_total = upper_clipped_mask_in_total.sum().item()
                                        lower_clipped_token_ratio_in_total_batch = lower_clipped_count_in_total / total_valid_token_count if total_valid_token_count > 0 else 0.0
                                        upper_clipped_token_ratio_in_total_batch = upper_clipped_count_in_total / total_valid_token_count if total_valid_token_count > 0 else 0.0

                                        template_ratio_metrics = {
                                            "actor/template_ratio/max": template_ratio_max,
                                            "actor/template_ratio/min": template_ratio_min,
                                            "actor/template_ratio/mean": template_ratio_mean,
                                            "actor/template_ratio/std": template_ratio_std,
                                            "actor/template_ratio/clipped_token_ratio_in_template_applied": clipped_ratio_in_template_applied,
                                            "actor/template_ratio/clipped_token_ratio_in_total_batch": clipped_ratio_in_total,
                                            "actor/template_ratio/lower_clipped_token_ratio_in_template_applied": lower_clipped_token_ratio_in_template_applied,
                                            "actor/template_ratio/upper_clipped_token_ratio_in_template_applied": upper_clipped_token_ratio_in_template_applied,
                                            "actor/template_ratio/lower_clipped_token_ratio_in_total_batch": lower_clipped_token_ratio_in_total_batch,
                                            "actor/template_ratio/upper_clipped_token_ratio_in_total_batch": upper_clipped_token_ratio_in_total_batch,
                                            "actor/template_ratio/cliprange_low": cliprange_low,
                                            "actor/template_ratio/cliprange_high": cliprange_high,
                                            "actor/template_ratio/soft_clip": 1.0 if soft_clip else 0.0,
                                        }
                                        metrics.update(template_ratio_metrics)

                                        print(f"[GEPA] Template ratio stats (template-applied only): max={template_ratio_max:.4f}, min={template_ratio_min:.4f}, "
                                              f"mean={template_ratio_mean:.4f}, std={template_ratio_std:.4f}")
                                        print(f"[GEPA] Clip stats in template-applied: lower_clip={lower_clipped_token_ratio_in_template_applied:.4f}, "
                                              f"upper_clip={upper_clipped_token_ratio_in_template_applied:.4f}, "
                                              f"total_clip={clipped_ratio_in_template_applied:.4f}")
                                        print(f"[GEPA] Clip stats in total batch: lower_clip={lower_clipped_token_ratio_in_total_batch:.4f}, "
                                              f"upper_clip={upper_clipped_token_ratio_in_total_batch:.4f}, "
                                              f"total_clip={clipped_ratio_in_total:.4f}")
                                    else:
                                        print(f"[GEPA] Warning: no valid tokens found for template ratio calculation")

                                    batch.batch["template_ratio_mask"] = template_ratio_mask

                                    print(f"[GEPA] Applied template_ratio_mask: {num_template_applied} template-applied rollouts processed, "
                                          f"clip mode={'soft' if soft_clip else 'hard'}, clip range=[{lower_bound:.2f}, {upper_bound:.2f}]")
                                else:
                                    print(f"[GEPA] No template-applied rollouts found, skipping template_ratio_mask calculation")

                    if self._gepa_enabled() and base_hard_mask is not None and self._gepa_cfg.get("replace_prompts_when_log_prob", False):
                        if not self._gepa_cfg.get("enable_template_ratio_mask", False):
                            batch = self._rebuild_batch_input(batch)

                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self._gepa_enabled():
                            self._collect_very_hard_samples(
                                batch=batch,
                                uid_to_sample=uid_to_sample_map,
                                rollouts_per_sample=self.config.actor_rollout_ref.rollout.n,
                            )


                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self._gepa_enabled() and base_hard_mask is not None and self._gepa_cfg.get("fix_importance_ratio", False):
                        batch = self._rebuild_batch_input(batch)

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        if (
                            self._gepa_enabled()
                            and self._gepa_cfg.get("drop_failed_template_rollout", False)
                            and "__gepa_template_applied__" in batch.non_tensor_batch
                        ):
                            template_applied = batch.non_tensor_batch["__gepa_template_applied__"]
                            token_level_rewards = batch.batch["token_level_rewards"]
                            sample_rewards = token_level_rewards.sum(dim=-1)

                            failed_template_mask = (template_applied) & (sample_rewards.cpu().numpy() == 0)
                            success_template_mask = (template_applied) & (sample_rewards.cpu().numpy() > 0)

                            num_dropped = failed_template_mask.sum()
                            num_success = success_template_mask.sum()
                            print(f"[drop_failed_template_rollout] Template rollouts: {num_success} success (reward>0), {num_dropped} failed (reward=0)")
                            if num_dropped > 0:
                                response_mask = batch.batch["response_mask"]
                                failed_template_mask_tensor = torch.from_numpy(failed_template_mask).to(
                                    response_mask.device
                                )
                                batch.batch["response_mask"] = response_mask * (~failed_template_mask_tensor).unsqueeze(-1)

                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
            self._finalize_epoch_prompt_optimization(epoch_idx=epoch)
