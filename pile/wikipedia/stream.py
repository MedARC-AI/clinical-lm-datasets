
import math
import os
import regex as re
import numpy as np

import boto3
import botocore
from datasets import concatenate_datasets, load_dataset


MIN_DOC_TOKENS = 50
NUM_SHARDS = 2

WIKI_DIR = '/weka/home-griffin/clinical_pile/wikipedia'
BUCKET_NAME = 'pile-everything-west' # replace with your bucket name


def download(s3, key):
    out_fn = os.path.join(WIKI_DIR, key.split('/')[-1])
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
    text = re.sub('\n{2,}', '\n\n', row['text'])
    text = re.sub(' +', ' ', text)
    text = re.sub('\t+', ' ', text)
    text = '# ' + text
    text = text.strip()
    return text


if __name__ == '__main__':
    shards = list(range(NUM_SHARDS))
    s3 = boto3.resource('s3')

    datasets = []

    for shard in shards:
        if shard < 10:
            suffix = '0' + str(shard)
        else:
            suffix = str(shard)
        
        key = f'dolma/v1/wiki-en-simple/en_simple_wiki-00{suffix}.json.gz'

        # Use boto3 to download the dataset
        shard_local_fn = download(s3, key)

        # Use HF to load it
        dataset = load_dataset('json', data_files=shard_local_fn, split='train')

        # Computing Number of Tokens / Document
        dataset = dataset.map(
            lambda row: {'num_tokens': len(re.split(r'\W+', row['text']))},
            num_proc=64
        )

        dataset = dataset.filter(lambda row: row['num_tokens'] >= MIN_DOC_TOKENS)

        dataset = dataset.map(
            lambda row: {'shard': shard, 'text': process(row), 'id': row['id']},
            remove_columns=['source', 'version']
        )

        datasets.append(dataset)

        # Remove shard
        print(f'Done with {shard_local_fn}. Removing it...')
        os.remove(shard_local_fn)

    datasets = concatenate_datasets(datasets)

    out_dir = os.path.join(WIKI_DIR, 'dataset_hf')
    total_toks = sum(datasets['num_tokens'])
    print(f'Saving {len(datasets)} Wikipedia articles with {total_toks} total tokens to {out_dir}')
    dataset.save_to_disk(out_dir)
