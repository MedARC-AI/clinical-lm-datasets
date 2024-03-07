#!/bin/bash
#SBATCH --account topmedarc
#SBATCH --partition a80
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --exclusive
#SBATCH --mem 400gb
#SBATCH --requeue

CONFIG=$1

source /etc/profile.d/modules.sh
module load cuda/12.1

export HF_HOME="/weka/home-griffin/cache/huggingface"
export TRANSFORMERS_CACHE="/weka/home-griffin/cache/huggingface/models"
export HF_DATASETS_CACHE="/weka/home-griffin/cache/huggingface/datasets/"

cd /weka/home-griffin/clinical-lm-datasets/ablations
srun train_axolotl.sh $CONFIG
