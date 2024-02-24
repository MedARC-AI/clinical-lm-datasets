#!/bin/bash
#SBATCH --account overtopmedarc
#SBATCH --partition=a80
#SBATCh --mem=400gb
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive

# Define the # of chunks (must line up with the above)
NUM_CHUNKS=8

cd /weka/home-griffin
source envs/data/bin/activate
cd clinical-lm-datasets

# Iterate over the range using a for loop
for ((i=1; i<=$NUM_CHUNKS; i++)); do
    echo "Starting Chunk $i of $NUM_CHUNKS"
    srun --gpus=1 --ntasks=1 --exclusive --mem=50gb python3 embed/main.py --batch_size 8 --chunk $i --num_chunks $NUM_CHUNKS &
done

wait
deactivate
