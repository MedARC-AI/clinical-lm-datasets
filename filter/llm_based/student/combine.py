from datasets import load_from_disk
from glob import glob

import os
import argparse
import multiprocess
from datasets import concatenate_datasets
import regex as re
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Filter')

    parser.add_argument('--dimension', default='quality')
    parser.add_argument('--pile_dir', default='/weka/home-griffin/clinical_pile/v1/dataset_hf_clean')
    parser.add_argument('--shard_dir', default='/weka/home-griffin/clinical_pile/v1/ask_llm_shards')
    parser.add_argument('--out_dir', default='/weka/home-griffin/clinical_pile/v1/ask_llm_hf')
    parser.add_argument('--excluded_sources', default='code|gutenberg_books')
    parser.add_argument('--min_doc_tokens', default=50, type=int)

    args = parser.parse_args()

    excluded_sources = set(args.excluded_sources.split('|'))

    shard_dirs = os.listdir(args.shard_dir)

    shard_idxs = []
    num_shards = []

    for dir in shard_dirs:
        match = re.search(r'(\d+)_(\d+)', dir)
        if match is None:
            print(dir)
        else:
            shard_idxs.append(int(match.group(1)))
            num_shards.append(int(match.group(2)))
    
    assert len(set(num_shards)) == 1

    num_shards = list(num_shards)[0]
    shard_idxs = list(sorted(shard_idxs))
    actual = len(shard_idxs)
    print(f'Expecting {num_shards} directories. Got {len(shard_idxs)}')

    missing_idxs = [i for i in range(num_shards) if i not in shard_idxs]

    if len(missing_idxs) > 0:
        print('Missing the following shards:')
        for missing in missing_idxs:
            print(f'\t- {missing}')
    else:
        datasets = []
        for shard_idx in tqdm(range(num_shards)):
            dir = os.path.join(args.shard_dir, f'{shard_idx}_{num_shards}')
            if len(os.listdir(dir)) == 0:
                print(f'{dir} is empty. This is deprecated behavior.')
            else:
                shard_dataset = load_from_disk(dir)
                # Filter out excluded sources (in future these will )
                shard_dataset = shard_dataset.filter(
                    lambda row: row['source'] not in excluded_sources, num_proc=multiprocess.cpu_count() - 8
                )
                datasets.append(shard_dataset)
        
        if len(excluded_sources) > 0:
            orig_data = load_from_disk(args.pile_dir)
            excluded_dataset = orig_data.filter(
                lambda row: row['source'] in excluded_sources, num_proc=multiprocess.cpu_count() - 8
            )
            datasets.append(excluded_dataset)

        datasets = concatenate_datasets(datasets)

        # Recalculate number of tokens
        datasets = datasets.map(
            lambda row: {'num_tokens_post_filter': len(re.split(r'\W+', row['text']))},
            num_proc=multiprocess.cpu_count() - 8
        )

        prev_n = len(datasets)
        prev_toks = sum(datasets['num_tokens'])

        datasets = datasets.filter(
            lambda row: row['num_tokens_post_filter'] >= args.min_doc_tokens,
            num_proc=multiprocess.cpu_count() - 8
        )

        new_n = len(datasets)
        new_toks = datasets['num_tokens_post_filter']

        print(f'Reduced token count from {prev_toks} to {new_toks}')
        print(f'Reduced documents from {prev_n} to {new_n}')

        print(f'Saving {len(datasets)} examples to {args.out_dir}')
        datasets.save_to_disk(args.out_dir)
