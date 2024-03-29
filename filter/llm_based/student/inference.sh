#!/bin/bash
#SBATCH --account medarc
#SBATCH --partition=h80i
#SBATCh --mem=200gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --output=inference_logs/O-%A-%a.out
#SBATCH --error=inference_logs/E-%A-%a.out

DIMENSION=$1
EXPERIMENT=$2
SHARDS=$3
SHARD_IDX=$((SLURM_ARRAY_TASK_ID - 1))

source /etc/profile.d/modules.sh
module load cuda/12.1

cd /weka/home-griffin
source envs/data/bin/activate
cd clinical-lm-datasets/filter/llm_based/student

python3 score_and_filter.py --dimension $DIMENSION --experiment $EXPERIMENT --shard_idx $SHARD_IDX --num_shards $SHARDS
