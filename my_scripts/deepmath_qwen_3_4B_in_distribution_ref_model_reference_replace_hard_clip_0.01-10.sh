#!/bin/bash

set -x

export RAY_ADDRESS="auto"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

export MODEL_PATH='xxx/Qwen3-4B'

export EXPERIMENT_NAME=deepmath_qwen_3_4B_in_distribution_ref_model_reference_replace_hard_clip_0.01-10

SAVE_PATH="./$EXPERIMENT_NAME"

# You need to manually set up a VLLM server as the reflection model
VLLM_SERVER_ADDR="http://localhost:13516/v1"  # Server address of qwen3 4b 

python3 -u -m verl.trainer.main_ppo_gepa_wo_additional_grpo_in_distribution \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/deepmath/baseline_boxed/very_hard_le_7/balanced/train_5000.parquet \
    data.val_files=./data/deepmath/baseline_boxed/very_hard_le_7/balanced/test_500.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=12288 \
    data.shuffle=true \
    reward_model.reward_manager=naive \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=600000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    trainer.critic_warmup=0 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.temperature=0.6 \
    trainer.logger=['wandb'] \
    trainer.val_only=false \
    trainer.val_before_train=true \
    trainer.project_name='grpo_with_gepa' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=10 \
    +trainer.gepa.enabled=True \
    +trainer.gepa.add_near_hard=True \
    +trainer.gepa.auto=heavy_4 \
    +trainer.gepa.dev_ratio=300 \
    +trainer.gepa.gepa_select_pareto_k=16 \
    +trainer.gepa.replace_prompts_when_log_prob=false \
    +trainer.gepa.fix_importance_ratio=false \
    +trainer.gepa.use_api_model_to_reflection=true \
    +trainer.gepa.enable_template_ratio_mask=true \
    +trainer.gepa.template_ratio_cliprange_low=0.99 \
    +trainer.gepa.template_ratio_cliprange_high=9 \
    +trainer.gepa.soft_clip=false \
    +trainer.gepa.api_type="openai" \
    +trainer.gepa.api_base=$VLLM_SERVER_ADDR \
    +trainer.gepa.model_name="qwen3-4b" \
    +trainer.gepa.thinking_in_reflection=false \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model']