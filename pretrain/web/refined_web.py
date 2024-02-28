
import math
import os
import regex as re
import numpy as np

import boto3
import botocore
from datasets import concatenate_datasets, load_dataset


MIN_DOC_TOKENS = 50

# Use RefinedWeb
WEB_DIR = '/weka/home-griffin/clinical_pile/refined_web'
os.makedirs(WEB_DIR, exist_ok=True)
BUCKET_NAME = 'pile-everything-west' # replace with your bucket name
TARGET_PAGES = 10000000


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
    return text


if __name__ == '__main__':


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
    pages_per_shard = math.ceil(TARGET_PAGES / num_shards)
    print(f'Getting {pages_per_shard} pages per shard (# {num_shards})')

    datasets = []
    for key in keys:
        shard = int(re.search(r'\d+', key).group())
        print(f'Starting with shard {shard}...')
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
            num_proc=64
        )

        sample = sample.filter(lambda row: row['num_tokens'] >= MIN_DOC_TOKENS)

        sample = sample.map(
            lambda row: {'shard': shard, 'text': process(row), 'id': row['url']},
            remove_columns=['image_urls', 'content']
        )

        datasets.append(sample)

        # Remove shard
        print(f'Done with {shard_local_fn}. Removing it...')
        os.remove(shard_local_fn)

    datasets = concatenate_datasets(datasets)

    out_dir = os.path.join(WEB_DIR, 'dataset_hf')
    total_toks = sum(datasets['num_tokens'])
    print(f'Saving {len(datasets)} web pages with {total_toks} total tokens to {out_dir}')
    dataset.save_to_disk(out_dir)

