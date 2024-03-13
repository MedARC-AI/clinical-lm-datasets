#!/bin/bash

set -e

NUM_CHUNKS=16

# Iterate over the range using a for loop
for ((i=1; i<=$NUM_CHUNKS; i++)); do
    echo "Submitting job to embed chunk $i of $NUM_CHUNKS"
    sbatch run_chunk.sh $i $NUM_CHUNKS
done
