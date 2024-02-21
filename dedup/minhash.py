import gzip
import json
import os
from collections import Counter, defaultdict
from glob import glob

import argparse
from datasets import Dataset, concatenate_datasets, load_dataset
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


HF_PATH = 'medarc/clinical_pile_v1'
MINHASH_BASE_PATH = '/weka/home-griffin/clinical_pile/v1/dedup/minhash'
os.makedirs(MINHASH_BASE_PATH, exist_ok=True)
SIG_DIR = os.path.join(MINHASH_BASE_PATH, 'signatures')
BUCKET_DIR = os.path.join(MINHASH_BASE_PATH, 'buckets')
CLUSTER_DIR = os.path.join(MINHASH_BASE_PATH, 'clusters')
REMOVED_IDS_DIR = os.path.join(MINHASH_BASE_PATH, 'removed_ids')
REMOVED_DIR = os.path.join(MINHASH_BASE_PATH, 'removed')
OUT_DIR = os.path.join(MINHASH_BASE_PATH, 'filtered_output')
TOTAL_TASKS = 32


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Min-Hash DeDup.')
    parser.add_argument('--keep_all_source_list', default='wikidoc', type=str)  # "|" comma delimited

    args = parser.parse_args()

    keep_all_source_set = set(args.keep_all_source_list.split('|'))

    # loader = HuggingFaceDatasetReader(
    #     dataset=HF_PATH,
    #     dataset_options={'split': 'train'},
    #     progress=True,
    #     text_key='text',
    #     id_key='id',
    # )
    # # you can also change ngrams or the number of buckets and their size here
    # minhash_config = MinhashConfig(use_64bit_hashes=True)  # better precision -> fewer false positives (collisions)

    # # this is the original data that we want to deduplicate
    # minhash_sig = MinhashDedupSignature(output_folder=SIG_DIR, config=minhash_config)

    # stage1 = LocalPipelineExecutor(
    #     pipeline=[loader, minhash_sig],
    #     logging_dir=os.path.join(SIG_DIR, 'logs'),
    #     tasks=TOTAL_TASKS,
    #     workers=32
    # )
    
    # minhash_buckets = MinhashDedupBuckets(
    #     input_folder=SIG_DIR,
    #     output_folder=BUCKET_DIR,
    #     config=minhash_config,
    # )

    # stage2 = LocalPipelineExecutor(
    #     pipeline=[minhash_buckets],
    #     logging_dir=os.path.join(BUCKET_DIR, 'logs'),
    #     tasks=minhash_config.num_buckets,
    #     workers=32,
    # )

    # minhash_cluster = MinhashDedupCluster(
    #     input_folder=BUCKET_DIR,
    #     output_folder=REMOVED_IDS_DIR,
    #     config=minhash_config,
    # )

    # stage3 = LocalPipelineExecutor(
    #     pipeline=[minhash_cluster],
    #     logging_dir=os.path.join(CLUSTER_DIR, 'logs'),
    #     tasks=1,  # What does this mean?
    #     workers=32,
    # )

    # minhash_filter = MinhashDedupFilter(
    #     input_folder=REMOVED_IDS_DIR,
    #     exclusion_writer=JsonlWriter(REMOVED_DIR),
    # )

    # token_counter = TokensCounter()
    # out_writer = JsonlWriter(output_folder=OUT_DIR)

    # stage4 = LocalPipelineExecutor(
    #     pipeline=[loader, token_counter, minhash_filter, out_writer],
    #     logging_dir=os.path.join(OUT_DIR, 'logs'),
    #     tasks=TOTAL_TASKS,
    #     workers=32
    # )

    # stage1.run()
    # stage2.run()
    # stage3.run()
    # stage4.run()

    # Save to disk and HF
    # Load JSONL
    unfiltered_dataset = load_dataset(HF_PATH, split='train')

    # assert len(unfiltered_dataset) == len(set(unfiltered_dataset['id']))

    id_counts = Counter(unfiltered_dataset['id'])

    print({k: v for k, v in id_counts.items() if v > 1})
    raise
    # print(len(set(unfiltered_dataset['id'])))
    # print(len(unfiltered_dataset))

    keep_ids = set()

    for fn in glob(os.path.join(OUT_DIR, '*.gz')):
        print(f'Extracting Remaining Documents from {fn}')
        with gzip.open(fn, 'r') as fd:
            lines = [l.strip() for l in fd.readlines() if len(l.strip()) > 0]
            lines = list(p_uimap(json.loads, lines))
            for line in lines:
                keep_ids.add(line['id'])

    def_keep_ids = set(unfiltered_dataset.filter(lambda row: row['source'] in keep_all_source_set)['id'])
    combined_keep_ids = def_keep_ids.union(keep_ids)

    n = len(unfiltered_dataset)
    filtered_dataset = unfiltered_dataset.filter(lambda row: row['id'] in combined_keep_ids)
    print(f'Keeping {n}/{len(filtered_dataset)} documents')

    hf_out_name = HF_PATH + '_minhash_deduped'
    filtered_dataset.save_to_hub(hf_out_name)
