#!/bin/bash
#SBATCH --account topmedarc
#SBATCH --partition a40
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --mem 200gb

source /etc/profile.d/modules.sh
module load cuda/12.1

EXPERIMENT=$1
PRETRAIN_WEIGHTS=$2

export HF_HOME="/weka/home-griffin/cache/huggingface"
export TRANSFORMERS_CACHE="/weka/home-griffin/cache/huggingface/models"
export HF_DATASETS_CACHE="/weka/home-griffin/cache/huggingface/datasets/"

cd /weka/home-griffin/clinical-lm-datasets/train
srun finetune.sh $EXPERIMENT $PRETRAIN_WEIGHTS
