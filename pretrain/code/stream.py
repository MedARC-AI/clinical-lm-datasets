
import math
import multiprocess
import shutil
import os
import regex as re
import numpy as np

import argparse
import boto3
import botocore
from datasets import concatenate_datasets, load_dataset, load_from_disk


MIN_DOC_TOKENS = 50

CODE_DIR = '/weka/home-griffin/clinical_pile/code'
SHARD_DIR = os.path.join(CODE_DIR, 'shards')
os.makedirs(SHARD_DIR, exist_ok=True)
BUCKET_NAME = 'pile-everything-west'
DUP_IDS = {
}


def download(s3, key):
    out_fn = os.path.join(SHARD_DIR, key.split('/')[-1])
    try:
        print(f'Downloading {key} from {BUCKET_NAME} and saving to {out_fn}')
        s3.Bucket(BUCKET_NAME).download_file(key, out_fn)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f'The object with key={key} does not exist on bucket={BUCKET_NAME}.')
        else:
            raise
    return out_fn


if __name__ == '__main__':
    parser =argparse.ArgumentParser('Subsample Python code.')
    
    parser.add_argument(
        '--target_num_files', default=100000, type=int, help='Total number of python files to sample.'
    )

    args = parser.parse_args()

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)

    keys = []
    # Iterate through objects in the bucket with the specified prefix
    for obj in bucket.objects.filter(Prefix='long_context_starcoder/python/'):
        if obj.key.endswith('.parquet'):
            keys.append(obj.key)

    num_shards = len(keys)
    files_per_shard = math.ceil(args.target_num_files / num_shards)
    print(f'Getting {files_per_shard} files per shard (# {num_shards})')

    shard_dirs = []
    for key in keys:
        shard = int(re.search(r'\d+', key).group())
        print(f'Starting with shard {shard}...')
        shard_dir = os.path.join(SHARD_DIR, f'{shard}_hf')
        shard_dirs.append(shard_dir)
        if shard < 10:
            suffix = '0' + str(shard)
        else:
            suffix = str(shard)

        if os.path.exists(shard_dir):
            print(f'Found existing processed shard at {shard_dir}. Skipping...')
        else:
            # Use boto3 to download the dataset
            shard_local_fn = download(s3, key)

            # Use HF to load it
            dataset = load_dataset('parquet', data_files=shard_local_fn, split='train').rename_column('content', 'text')

            idxs = np.arange(len(dataset))
            np.random.shuffle(idxs)

            sample = dataset.select(idxs[:files_per_shard])

            # Computing Number of Tokens / Document
            sample = sample.map(
                lambda row: {'num_tokens': len(re.split(r'\W+', row['text']))},
                num_proc=multiprocess.cpu_count()
            )

            sample = sample.filter(
                lambda row: row['num_tokens'] >= MIN_DOC_TOKENS,
                num_proc=multiprocess.cpu_count()
            )

            sample = sample.map(
                lambda row, idx: {
                    'id': row['max_stars_repo_name'] + '-shard=' + str(shard) + '-idx=' + str(idx),
                    'shard': shard,
            },
                num_proc=multiprocess.cpu_count(),
                with_indices=True
            )

            print(f'Saving {len(sample)} files to {shard_dir}')
            sample.save_to_disk(shard_dir)

            # Remove shard
            print(f'Done with {shard_local_fn}. Removing it...')
            os.remove(shard_local_fn)

    dataset = concatenate_datasets([load_from_disk(shard_dir) for shard_dir in shard_dirs])

    out_dir = os.path.join(CODE_DIR, 'dataset_hf')
    total_toks = sum(dataset['num_tokens'])
    print(f'Saving {len(dataset)} code files with {total_toks} total tokens to {out_dir}')
    dataset.save_to_disk(out_dir)

    for shard_dir in shard_dirs:
        print(f'Removing {shard_dir}')
        shutil.rmtree(shard_dir)
