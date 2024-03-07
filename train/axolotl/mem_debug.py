import psutil
import os
from datasets import load_from_disk, load_dataset, Dataset
import pyarrow
import numpy as np
import time
import h5py
from tqdm import tqdm
from p_tqdm import p_uimap


dataset = h5py.File('/weka/home-griffin/clinical_pile/v1/tokenized/dataset_hf_pubmed/0.h5', 'r')
input_ids = dataset.get('input_ids')
mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

for idx in range(len(input_ids)):
    mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    x = input_ids[idx]
    print(idx + 1, mem_before, mem_after - mem_before)

dataset.close()

shard_idx = 0

# def process_shard(shard_idx):
#     shard_in_fn = f'/weka/home-griffin/clinical_pile/v1/tokenized/dataset_hf_pubmed/data-0000{shard_idx}-of-00008.arrow'
#     print(f'Reading in {shard_in_fn}')
#     shard = Dataset.from_file(shard_in_fn)
#     shard = shard.select(list(range(10000)))

#     out_fn = f'/weka/home-griffin/debug_{shard_idx}.h5'
#     if os.path.exists(out_fn):
#         os.remove(out_fn)

#     start = time.time()
#     print('Starting count...')
#     with h5py.File(out_fn, 'w') as f:
#         out_data = f.create_dataset('input_ids', data=shard['input_ids'])
#     end = time.time()
#     print('Below is number of seconds it took')
#     print(end - start)
#     return 0

# list(p_uimap(process_shard, range(8), num_cpus=8))
