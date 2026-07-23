#!/bin/bash
set -e

# ============ 基础配置 ============
# MODEL="Qwen/Qwen3-4B"
MODEL='/ceph_home/arknet/hf_models/Qwen/Qwen3-4B'
HOST="0.0.0.0"
PORT=23167
SERVED_NAME="qwen3-4b"

# ============ GPU 配置 ============
export CUDA_VISIBLE_DEVICES=0
TP_SIZE=1

# ============ 启动服务 ============
vllm serve ${MODEL} \
    --served-model-name ${SERVED_NAME} \
    --host ${HOST} \
    --port ${PORT} \
    --tensor-parallel-size ${TP_SIZE} \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max_num_batched_tokens 65536 \
    --reasoning-parser deepseek_r1 \
    --trust-remote-code