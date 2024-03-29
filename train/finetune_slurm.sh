#!/bin/bash
#SBATCH --account topmedarc
#SBATCH -D .
#SBATCH --output=finetune_logs/O-%A-%a.out
#SBATCH --error=finetune_logs/E-%A-%a.out
#SBATCH --partition a40
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --exclusive
#SBATCH --mem 0

source /etc/profile.d/modules.sh
module load cuda/12.1

EXPERIMENT=$1
DATASET=$2
IS_COT=$3
BASE_MODEL=$4  # "Qwen/Qwen1.5-0.5B"
PRETRAIN_WEIGHTS=$5

export HF_HOME="/weka/home-griffin/cache/huggingface"
export HF_DATASETS_CACHE="/weka/home-griffin/cache/huggingface/datasets/"

cd /weka/home-griffin/clinical-lm-datasets/train
bash finetune.sh $EXPERIMENT $DATASET $IS_COT $BASE_MODEL $PRETRAIN_WEIGHTS
