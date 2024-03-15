#!/bin/bash

set -e

if [ "$1" = "topic" ]; then
    echo "Running topic filter..."
    sbatch --array=1-$2 run_topic.sh $2
elif [ "$1" = "quality" ]; then
    echo "Running quality filter..."
    sbatch --array=1-$2 run_quality.sh $2
else
    echo "Error: Bad argument. Please pass 'topic' or 'quality'." >&2
    exit 1
fi