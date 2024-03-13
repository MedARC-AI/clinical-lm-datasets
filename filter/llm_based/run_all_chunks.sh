#!/bin/bash

DIMENSION=$1
NUM_CHUNKS=$2

for ((i=0; i < $NUM_CHUNKS; i++)); do
    echo "Starting Chunk $i of $NUM_CHUNKS"
    sbatch run_chunk.sh $DIMENSION $i $NUM_CHUNKS
done
