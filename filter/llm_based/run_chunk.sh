#!/bin/bash
#SBATCH --account topmedarc
#SBATCH --partition=a80
#SBATCh --mem=200gb
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --exclusive

# Define the # of chunks (must line up with the above)
DIMENSION=$1
CHUNK=$2
NUM_CHUNKS=$3

source /etc/profile.d/modules.sh
module load cuda/12.1

cd /weka/home-griffin
source envs/data/bin/activate
cd clinical-lm-datasets

python3 filter/llm_based/gen_labels.py --dimension $DIMENSION --chunk $CHUNK --num_chunks $NUM_CHUNKS

wait
deactivate
