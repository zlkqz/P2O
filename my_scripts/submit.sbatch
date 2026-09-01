#!/bin/bash

#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH --qos=normal
#SBATCH -J p2o_training
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -e

echo "Current Work Dir: $(pwd)"

CONTAINER_NAME=verl_zhangkaiqi2024
WORK_DIR=/ceph_home/zhangkaiqi2024/P2O
SCRIPT_PATH=/ceph_home/zhangkaiqi2024/P2O/my_scripts/supplement/deepmath_10k_qwen3_4B_in_distribution_ref_model_reference_replace_hard_clip_0.01-10.sh

docker start ${CONTAINER_NAME}

cleanup() {
    echo "Stopping container ${CONTAINER_NAME}..."
    docker stop ${CONTAINER_NAME}
}
trap cleanup EXIT

docker exec -w ${WORK_DIR} ${CONTAINER_NAME} wandb login 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
docker exec -w ${WORK_DIR} ${CONTAINER_NAME} pip install math_verify

docker exec -w ${WORK_DIR} ${CONTAINER_NAME} bash ${SCRIPT_PATH}
