import gzip
import json
import multiprocess
import os
from collections import Counter
from glob import glob

import argparse
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from p_tqdm import p_uimap


MINHASH_BASE_PATH = '/weka/home-griffin/clinical_pile/v1/dedup/minhash'
os.makedirs(MINHASH_BASE_PATH, exist_ok=True)
SIG_DIR = os.path.join(MINHASH_BASE_PATH, 'signatures')
BUCKET_DIR = os.path.join(MINHASH_BASE_PATH, 'buckets')
CLUSTER_DIR = os.path.join(MINHASH_BASE_PATH, 'clusters')
REMOVED_IDS_DIR = os.path.join(MINHASH_BASE_PATH, 'removed_ids')
REMOVED_DIR = os.path.join(MINHASH_BASE_PATH, 'removed')
OUT_DIR = os.path.join(MINHASH_BASE_PATH, 'filtered_output')
TOTAL_TASKS = 1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Min-Hash LSH De-Duplication.')
    parser.add_argument('--pile_path', default='/weka/home-griffin/clinical_pile/v1/dataset_hf')
    parser.add_argument('--keep_all_source_list', default='wikidoc', type=str)  # "|" comma delimited
    parser.add_argument('--mode', default='all', choices=['all', 'dedup', 'save'])

    args = parser.parse_args()

    keep_all_source_set = set(args.keep_all_source_list.split('|'))

    if args.mode in {'all', 'dedup'}:
        loader = HuggingFaceDatasetReader(
            dataset=args.pile_path,
            local=True,
            # dataset_options={'split': 'train'},
            progress=True,
            text_key='text',
            # TODO change this to "uuid"
            id_key='id',
        )
        # you can also change ngrams or the number of buckets and their size here
        minhash_config = MinhashConfig(use_64bit_hashes=True)  # better precision -> fewer false positives (collisions)

        # this is the original data that we want to deduplicate
        minhash_sig = MinhashDedupSignature(output_folder=SIG_DIR, config=minhash_config)

        stage1 = LocalPipelineExecutor(
            pipeline=[loader, minhash_sig],
            logging_dir=os.path.join(SIG_DIR, 'logs'),
            tasks=TOTAL_TASKS,
            workers=multiprocess.cpu_count(),
        )

        minhash_buckets = MinhashDedupBuckets(
            input_folder=SIG_DIR,
            output_folder=BUCKET_DIR,
            config=minhash_config,
        )

        stage2 = LocalPipelineExecutor(
            pipeline=[minhash_buckets],
            logging_dir=os.path.join(BUCKET_DIR, 'logs'),
            tasks=minhash_config.num_buckets,
            workers=multiprocess.cpu_count()
        )

        minhash_cluster = MinhashDedupCluster(
            input_folder=BUCKET_DIR,
            output_folder=REMOVED_IDS_DIR,
            config=minhash_config,
        )

        stage3 = LocalPipelineExecutor(
            pipeline=[minhash_cluster],
            logging_dir=os.path.join(CLUSTER_DIR, 'logs'),
            tasks=1,  # What does this mean?
            workers=multiprocess.cpu_count(),
        )

        minhash_filter = MinhashDedupFilter(
            input_folder=REMOVED_IDS_DIR,
            exclusion_writer=JsonlWriter(REMOVED_DIR),
        )

        token_counter = TokensCounter()
        # out_writer = JsonlWriter(output_folder=OUT_DIR)

        # Compute before and after token counts and log
        stage4 = LocalPipelineExecutor(
            pipeline=[loader, token_counter, minhash_filter, token_counter],
            logging_dir=os.path.join(OUT_DIR, 'logs'),
            tasks=TOTAL_TASKS,
            workers=multiprocess.cpu_count(),
        )

        stage1.run()
        stage2.run()
        stage3.run()
        stage4.run()

    if args.mode in {'all', 'save'}:
        # Save HF HUB
        print(f'Loading PILE from {args.pile_path}')
        unfiltered_dataset = load_from_disk(args.pile_path)

        # 'pes2o' and 'wikipedia' have 7052 overlapping ids. Change how we form IDs for these datasets
        # assert len(unfiltered_dataset) == len(set(unfiltered_dataset['uuid']))

        print(f'Loading in the min-hash "removed" IDs from {OUT_DIR}')

        ids_to_remove = set()
        num_removed_guidelines = 0
        for fn in glob(os.path.join(REMOVED_DIR, '*.gz')):
            print(f'Extracting removed document IDs from {fn}')
            with gzip.open(fn, 'r') as fd:
                lines = fd.readlines()
                for line in tqdm(lines, total=len(lines)):
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    line = json.loads(line)
                    assert type(line) == dict
                    ids_to_remove.add(line['id'])

        n = len(unfiltered_dataset)
        # TODO change to UUID
        # Either it's been saved by min-hash deduping or it's one of our "always save all" keep_all_source_set category
        print(f'Keep each document if either:')
        print(f'- ID is not one of the {len(ids_to_remove)} documents to be removed.')
        print(f'- or source is in an always save category ({args.keep_all_source_list.split}).')
        filtered_dataset = unfiltered_dataset.filter(
            lambda row: row['id'] not in ids_to_remove or row['source'] in keep_all_source_set,
            num_proc=multiprocess.cpu_count()
        )

        print(f'Keeping {n}/{len(filtered_dataset)} documents')
        prev_tokens = sum(unfiltered_dataset['num_tokens'])
        new_tokens = sum(filtered_dataset['num_tokens'])
        print(f'We have removed {prev_tokens - new_tokens} / {prev_tokens}')

        hf_out_name = args.pile_path + '_minhash'
        print(f'Saving {hf_out_name} filtered examples to {hf_out_name}')
        filtered_dataset.save_to_disk(hf_out_name)
