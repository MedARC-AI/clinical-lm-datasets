

import os
import regex as re
import tempfile

import argparse
import h5py
import numpy as np
import pandas as pd
from autofaiss import build_index
from datasets import load_from_disk


tempfile.tempdir = '/weka/home-griffin/cache/tmp'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('combine all embedding files into one very large numpy array')

    parser.add_argument('--base_dir', default='/weka/home-griffin/clinical_pile/v1')

    args = parser.parse_args()

    embedding_dir = os.path.join(args.base_dir, 'embeddings')
    out_dir = os.path.join(args.base_dir, 'embedding_index')
    os.makedirs(out_dir, exist_ok=True)
    index_fn = os.path.join(out_dir, 'knn.index')
    uuid_fn = os.path.join(out_dir, 'uuids.txt')

    subdirs = os.listdir(embedding_dir)
    subdirs = list(sorted(subdirs, key=lambda x: int(x.split('-')[0])))

    ids = []
    embeddings = []

    for subdir in subdirs:
        full_subdir = os.path.join(embedding_dir, subdir)
        chunk_dirs = [x for x in os.listdir(full_subdir) if x != 'dataset_info.json']
        chunk_dirs = list(sorted(chunk_dirs, key=lambda x: int(x.split('-')[0])))
        assert int(subdir.split('-')[-1]) == len(subdirs)

        for chunk_dir in chunk_dirs:
            embed_fn = os.path.join(full_subdir, chunk_dir, 'embeddings.h5')
            data_dir = os.path.join(full_subdir, chunk_dir, 'ids')
            h5f = h5py.File(embed_fn,'r')
            embeddings.append(np.array(h5f.get('array')))
            ids += load_from_disk(data_dir)['uuid']
    
    print('Concatenating embeddings...')
    embeddings = np.concatenate(embeddings)

    build_index(
        embeddings=embeddings, file_format='npy', index_path=index_fn, save_on_disk=True, use_gpu=False,
        min_nearest_neighbors_to_retrieve=10,
        max_index_memory_usage='250gb',
        current_memory_available='500gb',
        max_index_query_time_ms=20,
        # make_direct_map=True  # Allows us to recreate the indices
    )

    with open(uuid_fn, 'w') as fd:
        fd.write('\n'.join(ids))

    # Make sure no duplicates and our ids array matches size of index
    assert len(ids) == len(set(ids)) == len(embeddings)