
import multiprocess
import numpy as np
from collections import defaultdict
from tqdm import tqdm
np.random.seed(1992)
import os
import pandas as pd
import itertools


def sample_dataset(dataset, reweighting_config, target_num_tokens):
    sources = set(dataset['source'])
    cwd = os.path.dirname(os.path.abspath(__file__))
    mixtures = pd.read_csv(os.path.join(cwd, 'mixtures', f'{reweighting_config}.csv'))
    keep_prob_by_source = dict(zip(mixtures['source'], mixtures['weight']))

    # Assert our re-weighting config naming matches the naming convention from the dataset
    for k in keep_prob_by_source:
        assert k in sources

    for source in sources:
        if source not in keep_prob_by_source:
            print(f'We are assuming that {source} has 0 weight...')
            keep_prob_by_source[source] = 0.0

    def sample_number_to_include(keep_prob):
        if keep_prob == 0:
            return 0
        elif keep_prob < 1:
            return 1 if np.random.random() < keep_prob else 0
        else:
            return int(keep_prob)

    source_arr = dataset['source']
    N = len(dataset)

    print('Sampling dataset indices according to re-weighting config...')
    keep_idxs = list(itertools.chain(
        *list(map(lambda i: [i] * sample_number_to_include(keep_prob_by_source[source_arr[i]]), tqdm(range(N), total=N)))
    ))

    # Shuffle the order
    print(f'Shuffling the order of the {len(keep_idxs)} indices.')
    np.random.shuffle(keep_idxs)

    num_tokens_arr = dataset['num_tokens']
    new_num_toks = sum([num_tokens_arr[i] for i in keep_idxs])

    if new_num_toks > target_num_tokens:
        frac_to_keep = target_num_tokens / new_num_toks
        num_to_keep = int(frac_to_keep * len(keep_idxs))
        print(f'Keeping {num_to_keep} / {len(keep_idxs)} to reduce token count from {new_num_toks} to ~ {target_num_tokens}')
        keep_idxs = keep_idxs[:num_to_keep]

    dataset = dataset.select(keep_idxs)

    print('Converting to Pandas DataFrame to record mixtures with groupby...')
    source_stats = dataset.remove_columns(['id', 'uuid', 'text']).to_pandas().groupby('source')
    stats = []
    for source, df in source_stats:
        stats.append({
            'source': source,
            'tokens': df['num_tokens'].sum(),
            'documents': len(df),
        })

    print('Done sampling and recording stats. Here\'s the final mixture...')
    stats = pd.DataFrame(stats)
    stats = stats.sort_values(by='tokens', ascending=False)
    print(stats.head(len(stats)))
    return dataset, stats
