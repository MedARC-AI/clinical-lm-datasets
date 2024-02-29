
import math
import multiprocess
import os
import regex as re
from collections import Counter

import argparse
import boto3
import botocore
import numpy as np
from datasets import concatenate_datasets, load_dataset

MIN_DOC_TOKENS = 50

# Use RefinedWeb
WEB_DIR = '/weka/home-griffin/clinical_pile/refined_web'
SHARD_DIR = os.path.join(WEB_DIR, 'shards')
os.makedirs(SHARD_DIR, exist_ok=True)
BUCKET_NAME = 'pile-everything-west' # replace with your bucket name


def download(s3, key):
    out_fn = os.path.join(WEB_DIR, key.split('/')[-1])
    try:
        print(f'Downloading {key} from {BUCKET_NAME} and saving to {out_fn}')
        s3.Bucket(BUCKET_NAME).download_file(key, out_fn)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f'The object with key={key} does not exist on bucket={BUCKET_NAME}.')
        else:
            raise
    return out_fn


def process(row):
    text = re.sub('\n{2,}', '\n\n', row['content'])
    text = re.sub(' +', ' ', text)
    text = re.sub('\t+', ' ', text)
    text = text.strip()
    text = '\n'.join([x.strip() for x in text.split('\n')])
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Refined Web Download from S3 and sample for target token count.')

    parser.add_argument('--target_num_pages', default=10000000, type=int, help='Total number of webpages to sample.')

    parser.add_argument('--chunk', default=None, type=int)
    parser.add_argument('--num_chunks', default=10, type=int)

    args = parser.parse_args()

    from datasets import load_from_disk

    s3 = boto3.resource('s3')
    key = f'falcon-refinedweb/'

    # List objects in the bucket with the specified prefix
    bucket = s3.Bucket(BUCKET_NAME)

    keys = []
    # Iterate through objects in the bucket with the specified prefix
    for obj in bucket.objects.filter(Prefix=key):
        if obj.key.endswith('.parquet'):
            keys.append(obj.key)

    num_shards = len(keys)
    pages_per_shard = math.ceil(args.target_num_pages / num_shards)
    print(f'Getting {pages_per_shard} pages per shard (# {num_shards})')

    if args.chunk is not None:
        keys = np.array_split(keys, args.num_chunks)[args.chunk]

    for key in keys:
        shard = int(re.search(r'\d+', key).group())

        print(f'Starting with shard {shard}...')
        shard_dir = os.path.join(SHARD_DIR, f'{shard}_hf')

        if os.path.exists(shard_dir):
            print(f'Found existing processed shard at {shard_dir}. Skipping...')
        else:
            # Use boto3 to download the dataset
            shard_local_fn = download(s3, key)

            # Use HF to load it
            dataset = load_dataset('parquet', data_files=shard_local_fn, split='train')

            idxs = np.arange(len(dataset))
            np.random.shuffle(idxs)

            sample = dataset.select(idxs[:pages_per_shard])

            # Computing Number of Tokens / Document
            sample = sample.map(
                lambda row: {'num_tokens': len(re.split(r'\W+', row['content']))},
                num_proc=multiprocess.cpu_count()
            )

            sample = sample.filter(
                lambda row: row['num_tokens'] >= MIN_DOC_TOKENS,
                num_proc=multiprocess.cpu_count()
            )

            sample = sample.map(
                lambda row: {'shard': shard, 'text': process(row), 'id': row['url']},
                remove_columns=['image_urls', 'content'],
                num_proc=multiprocess.cpu_count()
            )

            print(f'Saving {len(sample)} pages to {shard_dir}')
            sample.save_to_disk(shard_dir)

            # Avoid out of disk errors
            print('Cleaned up ' + str(sample.cleanup_cache_files()) + ' cache files')

            # Remove shard
            print(f'Done with {shard_local_fn}. Removing it...')
            os.remove(shard_local_fn)

    datasets = []
    for subdir in os.listdir(SHARD_DIR):
        full_path = os.path.join(SHARD_DIR, subdir)
        print(f'Adding {full_path} to full dataset')
        datasets.append(load_from_disk(full_path))

    datasets = concatenate_datasets(datasets)

    dup_ids = set([id[0] for id in Counter(datasets['id']).most_common() if id[1] > 1])

    print(f'Found {len(dup_ids)} duplicate URLs. Removing all instances')
    prev_n = len(datasets)
    datasets = datasets.filter(
        lambda row: row['id'] not in dup_ids,
        num_proc=multiprocess.cpu_count()
    )
    new_n = len(datasets)
    print(f'Removed {prev_n - new_n} web pages with duplicate URLs.')

    out_dir = os.path.join(WEB_DIR, 'dataset_hf')
    total_toks = sum(datasets['num_tokens'])
    print(f'Saving {len(datasets)} web pages with {total_toks} total tokens to {out_dir}')
    dataset.save_to_disk(out_dir)
