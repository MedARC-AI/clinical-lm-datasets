

import os
import regex as re

import argparse
import h5py
import numpy as np
import pandas as pd
from autofaiss import build_index
from datasets import load_from_disk


if __name__ == '__main__':
    parser = argparse.ArgumentParser('combine all embedding files into one very large numpy array')

    parser.add_argument('--embed_dir', default='/weka/home-griffin/clinical_pile/v1')
    parser.add_argument('--out_dir', default='/weka/home-griffin/clinical_pile/v1/embed_index')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args.index_fn = os.path.join(args.out_dir, 'knn.index')
    args.id_fn = os.path.join(args.out_dir, 'ids.txt')

    subdirs = os.listdir(args.embed_dir)

    subdirs = [d for d in subdirs if re.match(r'embed_\d+-\d+', d)]

    num_chunks = len(subdirs)
    assert num_chunks == int(subdirs[-1][-1])

    order = list(np.argsort([int(re.match(r'embed_(\d+)-\d+', d).group(1)) for d in subdirs]))

    subdirs = [subdirs[i] for i in order]

    print(subdirs)

    ids = []
    embeddings = []

    for dir in subdirs:
        embed_fn = os.path.join(args.embed_dir, dir, 'embeddings.h5')
        data_dir = os.path.join(args.embed_dir, dir, 'dataset_copy_hf')
        h5f = h5py.File(embed_fn,'r')
        embeddings.append(np.array(h5f.get('array')))
        ids += load_from_disk(data_dir)['id']

    print('Concatenating embeddings...')
    embeddings = np.concatenate(embeddings)

    build_index(
        embeddings=embeddings, file_format='npy', index_path=args.index_fn, save_on_disk=True, use_gpu=False,
        min_nearest_neighbors_to_retrieve=10,
        max_index_memory_usage='100gb',
        current_memory_available='250gb',
        max_index_query_time_ms=20,
        make_direct_map=True  # Allows us to recreate the indices
    )

    with open(args.id_fn, 'w') as fd:
        fd.write('\n'.join(ids))
