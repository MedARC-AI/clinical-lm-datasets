#!/bin/bash
#SBATCH --account topmedarc
#SBATCH --job-name pubmed_light_general
#SBATCH --partition a80
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --exclusive
#SBATCH --mem 300gb
#SBATCH --requeue

source /etc/profile.d/modules.sh
module load cuda/12.1

CONFIG=$1
EXPERIMENT=$2

export HF_HOME="/weka/home-griffin/cache/huggingface"
export TRANSFORMERS_CACHE="/weka/home-griffin/cache/huggingface/models"
export HF_DATASETS_CACHE="/weka/home-griffin/cache/huggingface/datasets/"

cd /weka/home-griffin/clinical-lm-datasets/ablations
srun train.sh $CONFIG $EXPERIMENT
