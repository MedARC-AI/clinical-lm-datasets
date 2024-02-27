from collections import Counter
import os
import regex as re

import argparse
from datasets import load_dataset


EXCLUDED_SOURCES = {
    'c4',
    'common-crawl',
    'reddit',
    'stack-dedup',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sample fraction of Dolma')

    parser.add_argument('--dolma_subset', default='v1_6')
    parser.add_argument('--target_tokens', default=1e9, type=int)  # 1 Billion

    args = parser.parse_args()

    dataset = load_dataset('allenai/dolma', args.dolma_subset, split='train')
    
    # Exclude Web Data and reddit
    dataset = dataset.filter(lambda row: row['source'] not in EXCLUDED_SOURCES)

    source_counts = Counter(dataset['source'])

    n = len(dataset)
    print(f'{n} total documents.')
    for k, ct in source_counts.most_common():
        print(k, ct, ct / n)

    dataset = dataset.map(
        lambda row: {'num_tokens': len(re.split(r'\W+', row['text']))},
        num_proc = 64
    )

    all_toks = sum(dataset['num_tokens'])

    print(f'\n\n{all_toks} total tokens.')

    for source in source_counts:
        d = dataset.filter(lambda row: row['source'] == source)
        ct = sum(d['num_tokens'])
        print(source, ct, ct / all_toks)
