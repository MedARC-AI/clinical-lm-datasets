

import gzip
import json
import multiprocess
import os
import itertools
from collections import Counter, defaultdict
from glob import glob

import argparse
from nltk import sent_tokenize
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup.sentence_dedup import SentenceDedupFilter, SentenceDedupSignature, SentenceFindDedups
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from p_tqdm import p_uimap
from tqdm import tqdm


SENT_DEDUP_DIR = '/weka/home-griffin/clinical_pile/v1/dedup/sentence'
OUT_FN = '/weka/home-griffin/clinical_pile/v1/dedup/sentence/dataset.jsonl'
os.makedirs(SENT_DEDUP_DIR, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sentence Level Exact Match Dedup.')
    parser.add_argument('--pile_path', default='/weka/home-griffin/clinical_pile/v1/dataset_hf_minhash')
    # Books are too long and will break the hashing signature on sentencel level
    # Code won't make sense if you filter at document level
    parser.add_argument('--excluded_sources', default='gutenberg_books|code')

    args = parser.parse_args()

    excluded_sources = set(args.excluded_sources.split('|'))

    # Need fewer than 312,748 sentences
    # print(f'Excluding ', excluded_sources)
    # def filter_func(row, to_remove=excluded_sources):
    #     return row['source'] not in to_remove and len(sent_tokenize(row['text'])) < 10000

    # loader = HuggingFaceDatasetReader(
    #     dataset=args.pile_path,
    #     # dataset_options={'split': 'train'},
    #     progress=True,
    #     filter_func=filter_func,
    #     local=True,
    #     text_key='text',
    #     id_key='uuid',
    # )

    # step1 = LocalPipelineExecutor(
    #     pipeline=[loader, SentenceDedupSignature(output_folder=SENT_DEDUP_DIR)],
    #     logging_dir=os.path.join(SENT_DEDUP_DIR, 'sig_logs'),
    #     tasks=4,
    #     workers=4
    # )
    
    # step2 = LocalPipelineExecutor(
    #     pipeline=[SentenceFindDedups(data_folder=SENT_DEDUP_DIR, output_folder=SENT_DEDUP_DIR)],
    #     logging_dir=os.path.join(SENT_DEDUP_DIR, 'find_logs'),
    #     tasks=1,
    #     workers=1
    # )

    # exclusion_writer = JsonlWriter(output_folder=os.path.join(SENT_DEDUP_DIR, 'removed_sentences'))
    # step3 = LocalPipelineExecutor(
    #     pipeline=[
    #         loader,
    #         SentenceDedupFilter(data_folder=SENT_DEDUP_DIR, exclusion_writer=exclusion_writer),
    #         JsonlWriter(output_folder=os.path.join(SENT_DEDUP_DIR, 'filtered_output'))
    #     ],
    #     logging_dir=os.path.join(SENT_DEDUP_DIR, 'filter_logs'),
    #     tasks=4,
    #     workers=4
    # )

    # print(step1.run())
    # print(step2.run())
    # print(step3.run())

    in_fns = list(glob(
        os.path.join(SENT_DEDUP_DIR, 'filtered_output', '*jsonl.gz'))
    )

    print(in_fns)
    # def process_chunk(fn):
    #     out = []
    #     print(f'Loading {fn}...')
    #     with gzip.open(fn, 'rt', encoding='utf-8') as f:
    #         for line in tqdm(f):
    #             try:
    #                 x = json.loads(line.strip())
    #                 out.append(x)
    #             except:
    #                 print(f'Could not decode JSON line --> {line}.')
    #     return out
    
    # dataset = Dataset.from_list(
    #     list(itertools.chain(*list(p_uimap(process_chunk, in_fns, num_cpus=len(in_fns)))))
    # )

    # dataset = list()
    invalid_json = 0
    saved = 0
    sources_seen = set()

    print(f'Writing outputs to {OUT_FN}. Removing ones which aren\'t valid JSON.')
    with open(OUT_FN, 'w', encoding='utf-8') as out_fd:
        for fn in in_fns:
            print(f'Loading {fn}...')
            with gzip.open(fn, 'rt', encoding='utf-8') as in_fd:
                for line in tqdm(in_fd):
                    line = line.strip()
                    x = json.loads(line)

                    # This was our UUID
                    uuid = x.pop('id')
                    x['metadata'].pop('dataset')

                    new_obj = {
                        'text': x['text'],
                        'uuid': uuid,
                    }

                    new_obj.update(x['metadata'])

                    # ['text', 'num_tokens', 'id', 'source', 'meta', 'dataset']
                    keys_needed = [
                        'num_tokens', 'id', 'text', 'source', 'uuid', 'meta'
                    ]

                    for key in keys_needed:
                        assert key in new_obj

                    sources_seen.add(new_obj['source'])
                    out_fd.write(json.dumps(new_obj) + '\n')
                    saved += 1

                    if saved % 1000000 == 0:
                        print(f'{saved} written so far to {OUT_FN}')

        if len(excluded_sources) > 0:
            assert len(excluded_sources.intersection(sources_seen)) == 0
            original = load_from_disk(args.pile_path)

            excluded_data = original.filter(
                lambda row: row['source'] in excluded_sources,
                num_proc=multiprocess.cpu_count()
            )

            print(f'Dumping {len(excluded_data)} files that we excluded from sentence-level de-duping.')

            for row in tqdm(excluded_data):
                out_fd.write(json.dumps(row) + '\n')
        
    print(f'{invalid_json} documents were saved by sentence-level de-duping which were not JSON de-serializable. These were removed')
    # dataset = Dataset.from_list(dataset)
    # datasets = [dataset]
    # datasets = []
    # for fn in tqdm(out_fns):
    #     dataset = load_dataset('json', data_files=fn, split='train')
    #     datasets.append(dataset)

        # datasets.append(excluded_data)

    # dataset = concatenate_datasets(datasets)

    # out_dir = SENT_DEDUP_DIR + '_hf'
    # print(f'Saving to {out_dir}')
    # dataset.save_to_disk(out_dir)

    # TODO
    # Remove temporary gz files
