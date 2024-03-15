#!/bin/bash
# Wrapper script to submit a job with dynamic GPU allocation

# Check if the user has provided a number of GPUs
if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_gpus>"
  exit 1
fi

NUM_CHUNKS=$1

# Submit the job with the specified number of GPUs
sbatch --array=1-$NUM_CHUNKS run_chunks_slurm.sh $NUM_CHUNKS
