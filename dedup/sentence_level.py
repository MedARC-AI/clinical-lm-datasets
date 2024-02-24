

import gzip
import json
import os
from collections import Counter, defaultdict
from glob import glob

import argparse
from datasets import Dataset, concatenate_datasets, load_dataset
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup.sentence_dedup import SentenceDedupFilter, SentenceDedupSignature, SentenceFindDedups
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from p_tqdm import p_uimap


HF_PATH = 'medarc/clinical_pile_v1_minhash_deduped'
SENT_DEDUP_DIR = '/weka/home-griffin/clinical_pile/v1/dedup/sentence'
os.makedirs(SENT_DEDUP_DIR, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sentence Level Exact Match Dedup.')

    args = parser.parse_args()

    loader = HuggingFaceDatasetReader(
        dataset=HF_PATH,
        dataset_options={'split': 'train'},
        progress=True,
        text_key='text',
        id_key='id',
    )

    step1 = LocalPipelineExecutor(
        pipeline=[loader, SentenceDedupSignature(output_folder=SENT_DEDUP_DIR)],
        logging_dir=os.path.join(SENT_DEDUP_DIR, 'sig_logs'),
        tasks=4,
        workers=4
    )
    
    step2 = LocalPipelineExecutor(
        pipeline=[SentenceFindDedups(data_folder=SENT_DEDUP_DIR, output_folder=SENT_DEDUP_DIR)],
        logging_dir=os.path.join(SENT_DEDUP_DIR, 'find_logs'),
        tasks=1,
        workers=1
    )
   
    exclusion_writer = JsonlWriter(output_folder=os.path.join(SENT_DEDUP_DIR, 'removed_sentences'))
    step3 = LocalPipelineExecutor(
        pipeline=[
            loader,
            SentenceDedupFilter(data_folder=SENT_DEDUP_DIR, exclusion_writer=exclusion_writer),
            JsonlWriter(output_folder=os.path.join(SENT_DEDUP_DIR, 'filtered_output'))
        ],
        logging_dir=os.path.join(SENT_DEDUP_DIR, 'filter_logs'),
        tasks=4,
        workers=4
    )

    # print(step1.run())
    # print(step2.run())
    print(step3.run())
