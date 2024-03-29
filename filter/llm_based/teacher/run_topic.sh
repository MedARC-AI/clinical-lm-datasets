#!/bin/bash
#SBATCH --account medarc
#SBATCH --partition=a40
#SBATCh --mem=200gb
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --job-name=ask_llm_chunks
#SBATCH --output=topic_logs/O-%A-%a.out
#SBATCH --error=topic_logs/E-%A-%a.out

# Define the # of chunks (must line up with the above)
DIMENSION=topic
NUM_CHUNKS=$1
CHUNK_IDX=$((SLURM_ARRAY_TASK_ID - 1))

source /etc/profile.d/modules.sh
module load cuda/12.1

cd /weka/home-griffin
source envs/data/bin/activate
cd clinical-lm-datasets

python3 filter/llm_based/gen_labels.py --dimension $DIMENSION --chunk $CHUNK_IDX --num_chunks $NUM_CHUNKS
