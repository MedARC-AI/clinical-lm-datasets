#!/bin/bash

set -e

python multimedqa.py
python mednli.py
python medalpaca.py
python chat_doctor.py

echo "Aggregating individual datasets into instruction pile"
python create_pile.py
