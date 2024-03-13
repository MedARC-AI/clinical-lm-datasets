#!/bin/bash
#SBATCH --account=medarc
#SBATCH --job-name=embed
#SBATCH --partition=a40
#SBATCH --mem=200gb
#SBATCH --gpus=1
#SBATCH --requeue

# Define the # of chunks (must line up with the above)
CHUNK=$1
NUM_CHUNKS=$2

source /etc/profile.d/modules.sh
module load cuda/12.1

cd /weka/home-griffin
source envs/data/bin/activate
cd clinical-lm-datasets

echo "Entering script to embed chunk $CHUNK of $NUM_CHUNKS"
python3 embed/main.py --batch_size 1 --chunk $CHUNK --num_chunks $NUM_CHUNKS
