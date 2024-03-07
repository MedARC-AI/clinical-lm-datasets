#!/bin/bash

set -e

CONFIG=$1
ACCELERATE_AXOLOTL_ABLATION_CONFIG=/weka/home-griffin/cache/huggingface/accelerate/axolotl_ablation_config.yaml

echo "Generating the ablation dataset if it does not exist..."
source /weka/home-griffin/envs/data/bin/activate
cd /weka/home-griffin/clinical-lm-datasets/tokenize
python3 build_dataset.py -axolotl --reweighting_config $CONFIG

cd /weka/home-griffin/clinical-lm-datasets/ablations
source /weka/home-griffin/envs/axolotl/bin/activate
cd /weka/home-griffin/axolotl

accelerate launch --config_file $ACCELERATE_AXOLOTL_ABLATION_CONFIG -m axolotl.cli.train "ablations_${CONFIG}.yml"
