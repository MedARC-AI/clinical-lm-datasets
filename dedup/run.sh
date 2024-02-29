#!/bin/bash

set -e

PILE_NAME=medarc/clinical_pile_v2

python3 minhash.py --mode all --pile_path $PILE_NAME
