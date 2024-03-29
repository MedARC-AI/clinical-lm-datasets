#!/bin/bash

set -e

DIMENSION=$1
EXPERIMENT=$2
NUM_SHARDS=$3

for ((SHARD_IDX=0; SHARD_IDX<$NUM_SHARDS; SHARD_IDX++)); do
    python3 score_and_filter.py --dimension $DIMENSION --experiment $EXPERIMENT --shard_idx $SHARD_IDX --num_shards $NUM_SHARDS
done
