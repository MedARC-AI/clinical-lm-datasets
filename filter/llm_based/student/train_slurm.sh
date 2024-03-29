#!/bin/bash
#SBATCH --account medarc
#SBATCH --partition=a40
#SBATCh --mem=200gb
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --job-name=ask_llm_train
#SBATCH --output=logs/O-%A-%a.out
#SBATCH --error=logs/E-%A-%a.out

set -e

DIMENSION=$1
EXPERIMENT=$2

source /etc/profile.d/modules.sh
module load cuda/12.1

cd /weka/home-griffin
source envs/data/bin/activate
cd clinical-lm-datasets/filter/llm_based

bash train.sh $DIMENSION $EXPERIMENT