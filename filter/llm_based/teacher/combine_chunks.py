import argparse
import os
import json

from glob import glob
from p_tqdm import p_uimap
from collections import Counter


def process(fn):
    with open(fn, 'r') as fd:
        return json.load(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Combining chunks for Mixtral-generated quality scores.')

    parser.add_argument('--pile_dir', default='/weka/home-griffin/clinical_pile/v1')
    parser.add_argument('--dimension', default='topic', choices=['topic', 'quality'])

    parser.add_argument('-save_combined', default=False, action='store_true')

    args = parser.parse_args()
    
    data_dir = os.path.join(args.pile_dir, f'dataset_hf_1mn_sample_llm_{args.dimension}_scores')
    fns = list(glob(os.path.join(data_dir, '*.json')))
    chunk_cts = Counter([int(x.split('/')[-1].split('_')[0]) for x in fns])

    min_chunk = min(chunk_cts)
    max_chunk = max(chunk_cts)

    for idx in range(min_chunk, max_chunk + 1):
        ct = chunk_cts.get(idx, 0)
        print(f'Chunk {idx} --> {ct}')

    arr = list(p_uimap(process, fns))

    uuids = [x['uuid'] for x in arr]
    print(len(uuids), len(set(uuids)))

