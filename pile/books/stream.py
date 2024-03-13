
import math
import os
import regex as re
import numpy as np

import boto3
import botocore
from datasets import concatenate_datasets, load_dataset


MIN_DOC_TOKENS = 50
NUM_SHARDS = 2


BOOKS_DIR = '/weka/home-griffin/clinical_pile/books'
BUCKET_NAME = 'pile-everything-west' # replace with your bucket name
DUP_IDS = {
    'e927dbe32bf2ccfa7a141fcf4c3ce145d7f73918',
    '61c83aeb7f8c23e65b207749db290b66a0ca393a',
    '8ea47781d1f36e5773e5f8b78576723052ca8dcc',
    '566f1eab849eb48788103eacccce342e80e74c23',
    'e44b38eca89b8f1bc4183e6b10da01f73aa67ebd',
    'fc25a4d62e71d68be396cc6c17c720a87b078b6e'
}


def download(s3, key):
    out_fn = os.path.join(BOOKS_DIR, key.split('/')[-1])
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
    text = '\n'.join([x.strip() for x in text.split('\n')])
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
        
        key = f'dolma/v1/gutenberg-books/books-00{suffix}.json.gz'

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

    out_dir = os.path.join(BOOKS_DIR, 'dataset_hf')
    total_toks = sum(datasets['num_tokens'])
    print(f'Saving {len(datasets)} books with {total_toks} total tokens to {out_dir}')
    dataset.save_to_disk(out_dir)
