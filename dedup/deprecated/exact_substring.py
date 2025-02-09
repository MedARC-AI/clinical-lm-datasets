import gzip
import json
import os
from collections import Counter, defaultdict
from glob import glob

import argparse
from datasets import Dataset, concatenate_datasets, load_dataset
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup.exact_substrings import ESDatasetToSequence, ESMergeSequences, ESRangeRemover
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from p_tqdm import p_uimap


SUBSTRING_BASE_PATH = '/weka/home-griffin/clinical_pile/v1/dedup/substring'
os.makedirs(SUBSTRING_BASE_PATH, exist_ok=True)
MERGE_DIR = os.path.join(SUBSTRING_BASE_PATH, 'merge')
REMOVE_DIR = os.path.join(SUBSTRING_BASE_PATH, 'remove')
OUT_DIR = os.path.join(SUBSTRING_BASE_PATH, 'filtered')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Min-Hash DeDup.')
    parser.add_argument('--pile_path', default='medarc/clinical_pile_v1_minhash_dedup', type='str')  # "|" comma delimited
    parser.add_argument('--stage', default=1, type=int, choices=[1, 2])

    args = parser.parse_args()

    include_all_sources = set(args.do_not_filter.split('|'))

    loader = HuggingFaceDatasetReader(
        dataset=args.pile_path,
        dataset_options={'split': 'train'},
        progress=True,
        text_key='text',
        id_key='id',
    )

    if args.stage == 1:
        stage1_a = LocalPipelineExecutor(
            pipeline=[loader, ESDatasetToSequence(output_folder=MERGE_DIR)],
            logging_dir=os.path.join(MERGE_DIR, 'logs_1a'),
            tasks=4,
            workers=4
        )

        print(stage1_a.run())

        merger = ESMergeSequences(
            data_folder=MERGE_DIR,
            tasks_stage_1=4,
        )

        stage1_b = LocalPipelineExecutor(
            pipeline=[merger],
            logging_dir=os.path.join(MERGE_DIR, 'logs_1b'),
            tasks=4,
            workers=4
        )

        print(stage1_b.run())
    else:
        remover = ESRangeRemover(sequence_folder=MERGE_DIR)
        writer = JsonlWriter(OUT_DIR)
        stage2 = LocalPipelineExecutor(
            pipeline=[remover, writer],
            logging_dir=os.path.join(MERGE_DIR, 'logs_2'),
            tasks=4,
            workers=4
        )

        filtered_dataset = load_dataset('json', data_files=os.path.join(OUT_DIR, '*.gz'), split='train')

        for fn in glob(os.path.join(OUT_DIR, '*.gz')):
            print(f'Extracting Remaining Documents from {fn}')
            with gzip.open(fn, 'r') as fd:
                lines = [l.strip() for l in fd.readlines() if len(l.strip()) > 0]
                lines = p_uimap(json.loads, lines)
                for line in lines:
                    if line['source'] in include_all_sources:
                        continue
                    else:
                        filtered_dataset.append(line)

        filtered_sources = set(filtered_dataset['source'])
        to_add = filtered_dataset.filter(lambda row: row['source'] in include_all_sources)
        to_add_sources = set(to_add['source'])
        assert len(to_add_sources.intersection(filtered_sources)) == 0

        print(num_removed)
        print(Counter(to_add['source']).most_common())

        final_dataset = concatenate_datasets([filtered_dataset, to_add])
        hf_out_name = args.pile_path + '_substring_dedup'
        final_dataset.save_to_hub(hf_out_name)
