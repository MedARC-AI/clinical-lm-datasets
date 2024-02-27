
import math
import os
import regex as re
import numpy as np

import boto3
import botocore
from datasets import concatenate_datasets, load_dataset


PUBMED_CORPUS_ID_FN = '/weka/home-griffin/clinical_pile/pubmed/s2orc/s2orc-PubMed_processed_corpusids.txt'
print(f'Loading all S2 Corpus IDs from our PubMed corpus to avoid re-using for replay PeS2o dataset.')
PUBMED_CORPUS_IDS = set([
    str(x.strip()) for x in open(PUBMED_CORPUS_ID_FN, 'r').readlines() if len(x.strip()) > 0
])
MIN_DOC_TOKENS = 50
NUM_SHARDS = 42
TARGET_ARTICLES = 1000000


PES2O_DIR = '/weka/home-griffin/clinical_pile/pes2o'
BUCKET_NAME = 'pile-everything-west' # replace with your bucket name


def download(s3, key, PES2O_DIR):
    out_fn = os.path.join(PES2O_DIR, key.split('/')[-1])
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
    title = row['metadata']['title']
    abstract = row['metadata']['abstract']
    sections = re.split('\n{2,}', row['text'])
    out_sections = [f'# {title}', f'## ABSTRACT\n{abstract}']

    for section in sections:
        new_sec = '\n'.join([
            '## ' + line if line.upper() == line else line for line in section.split('\n')
        ])
        out_sections.append('\n'.join(new_sec))
    out_str = '\n\n'.join(out_sections)
    out_str = re.sub(' +', ' ', out_str).strip()
    return out_str


if __name__ == '__main__':
    shards = list(range(NUM_SHARDS))
    articles_per_shard = math.ceil(TARGET_ARTICLES / NUM_SHARDS)

    s3 = boto3.resource('s3')

    datasets = []

    for shard in shards:
        if shard < 10:
            suffix = '0' + str(shard)
        else:
            suffix = str(shard)
        
        key = f'dolma/v1/peS2o/s2_v3-00{suffix}.json.gz'

        # Use boto3 to download the dataset
        shard_local_fn = download(s3, key, PES2O_DIR)

        # Use HF to load it
        dataset = load_dataset('json', data_files=shard_local_fn, split='train')
        # Filter out corpus ids in our pubmed dataset
        dataset = dataset.filter(lambda row: row['id'] not in PUBMED_CORPUS_IDS)

        # Computing Number of Tokens / Document
        dataset = dataset.map(
            lambda row: {'num_tokens': len(re.split(r'\W+', row['text']))},
            num_proc=64
        )

        dataset = dataset.filter(lambda row: row['num_tokens'] >= MIN_DOC_TOKENS)

        idxs = np.arange(len(dataset))
        np.random.shuffle(idxs)

        sample = dataset.select(idxs[:articles_per_shard])

        remove_cols = [
            col for col in list(sample.features) if col not in {'text', 'id'}
        ]

        sample = sample.map(
            lambda row: {'shard': shard, 'text': process(row), 's2_corpusid': row['id'], 'id': row['id']},
            remove_columns=remove_cols
        )

        datasets.append(sample)

        # Remove shard
        print(f'Done with {shard_local_fn}. Removing it...')
        os.remove(shard_local_fn)

    datasets = concatenate_datasets(datasets)

    out_dir = os.path.join(PES2O_DIR, 'dataset_sample_hf')
    print(f'Saving {len(datasets)} to {out_dir}')
    dataset.save_to_disk(out_dir)
