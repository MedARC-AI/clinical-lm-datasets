from collections import Counter

import os
import argparse
import math
import multiprocess
import pandas as pd
import numpy as np
from datasets import load_from_disk, concatenate_datasets
np.random.seed(1992)
from tqdm import tqdm


def shorten_number(num):
    suffixes = ['', 'k', 'mn', 'bn', 'tn']  # Add more suffixes as needed
    magnitude = 0
    while num >= 1000:
        num /= 1000.0
        magnitude += 1
    return '{}{}'.format(int(num), suffixes[magnitude])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--pile_dir', default='/weka/home-griffin/clinical_pile/v1')
    parser.add_argument('--target_examples', default=50000, type=int)
    parser.add_argument('--min_source_docs', default=None, type=int)

    args = parser.parse_args()

    target_no_str = shorten_number(args.target_examples)
    args.pile_path = os.path.join(args.pile_dir, 'dataset_hf')
    args.out_path = os.path.join(args.pile_dir, f'dataset_hf_{target_no_str}_sample')

    print(f'Loading dataset from {args.pile_path}...')
    print(f'Will save {target_no_str}-sized sample to {args.out_path}...')
    dataset = load_from_disk(args.pile_path)

    source_fn = os.path.join(args.pile_dir, 'sources.csv')
    if os.path.exists(source_fn):
        sources = pd.read_csv(source_fn)
        all_tokens = sources['tokens'].sum()

        # Find a target for each source
        keep_prob_by_source = {}
        for record in sources.to_dict('records'):
            source = record['source']
            tokens = record['tokens']
            ct = record['examples']

            sfrac = tokens / all_tokens
            print(f'Source={source} -> {round(sfrac * 100, 3)}% of corpus')
            sub_n = int(math.ceil(sfrac * args.target_examples))

            if args.min_source_docs:
                sub_n = max(sub_n, args.min_source_docs)
            
            if sub_n > ct:
                print(f'We do not have {sub_n} documents for {source}. Taking all ({ct}).')
                sub_n = ct

            sample_prob = min(1, sub_n / ct)
            print(f'Sampling {source} documents with {sample_prob} keep probability')
            keep_prob_by_source[source] = sample_prob

        def should_keep(row):
            keep_prob = keep_prob_by_source[row['source']]
            if keep_prob == 1:
                return True
            return np.random.random() < keep_prob
    
        filtered = dataset.filter(should_keep, num_proc=multiprocess.cpu_count())
        print(f'Saving {len(filtered)} to {args.out_path}...')
        print(len(filtered))
        filtered.save_to_disk(args.out_path)
    else:

        N = sum(dataset['num_tokens'])
        all_data = []
        print('Converting to Pandas then grouping by source...')
        for source, sdata in dataset.to_pandas().groupby('source'):
            # for source in tqdm(sources):
            # sdata = dataset.filter(
            #     lambda row: row['source'] == source,
            #     num_proc=multiprocess.cpu_count()
            # )
            num_tokens = sum(sdata['num_tokens'])
            ct = len(sdata)
            sfrac = num_tokens / N
            print(f'Source={source} -> {round(sfrac * 100, 3)}% of corpus')
            assert ct == len(sdata)

            sub_n = int(math.ceil(sfrac * args.target_examples))

            if args.min_source_docs:
                sub_n = max(sub_n, args.min_source_docs)
            
            if sub_n > ct:
                print(f'We do not have {sub_n} documents for {source}. Taking all ({ct}).')
                sub_n = ct

            # sidxs = np.arange(len(sdata))
            # np.random.shuffle(sidxs)

            # print(f'Sampling {sub_n} / {len(sdata)} from {source}')
            # chosen_idxs = sidxs[:sub_n]
            sdata_sample = sdata.sample(n=sub_n, replace=False, random_state=1992)

            # sdata_sample = sdata.select(chosen_idxs)
            all_data.append(sdata_sample)
        
        all_data = pd.concat(all_data)
        from datasets import Dataset
        all_data = Dataset.from_pandas(all_data)

        # all_data = concatenate_datasets(all_data)
        print(f'Saving {len(all_data)} to {args.out_path}...')
        all_data.save_to_disk(args.out_path)
