import os
import sys

v = os.environ['SLURM_ARRAY_TASK_ID']
print(f'Within File Env: {v}. Within File Passed In: {sys.argv[1]}')

import torch

print(torch.cuda.device_count())