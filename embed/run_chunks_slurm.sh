#!/bin/bash
#SBATCH --account=medarc
#SBATCH --job-name=run_embed_chunks
#SBATCH --output=logs/O-%A-%a.out
#SBATCH --error=logs/E-%A-%a.out
#SBATCH --partition=a40
#SBATCH --mem=100gb
#SBATCH --gpus=1
#SBATCH --requeue

NUM_CHUNKS=$1

source /etc/profile.d/modules.sh
module load cuda/12.1

cd /weka/home-griffin
source envs/data/bin/activate
cd clinical-lm-datasets

CHUNK_IDX=$((SLURM_ARRAY_TASK_ID))

echo "Processing Chunk ${CHUNK_IDX} / ${NUM_CHUNKS}"

python3 embed/main.py --chunk $CHUNK_IDX --num_chunks $NUM_CHUNKS --batch_size 1
