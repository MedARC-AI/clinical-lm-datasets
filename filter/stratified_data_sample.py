from collections import Counter
import pandas as pd
from datasets import load_dataset, concatenate_datasets
import numpy as np
np.random.seed(1992)

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--hf_path', default='medarc/clinical_pile_v1_minhash_deduped')
    parser.add_argument('--out_path', default='/weka/home-griffin/clinical_pile/v1/sample_hf')
    parser.add_argument('--target_examples', default=10000, type=int)

    args = parser.parse_args()

    dataset = load_dataset(args.hf_path, split='train')

    all_data = []
    for source, ct in Counter(dataset['source']).most_common():
        sdata = dataset.filter(lambda row: row['source'] == source)
        assert ct == len(sdata)
        frac_mixture = ct / len(dataset)
        sub_n = min(max(1, int(round(frac_mixture * args.target_examples))), ct)

        sidxs = np.arange(len(sdata))
        np.random.shuffle(sidxs)

        print(f'Sampling {sub_n} / {len(sdata)} from {source}')
        chosen_idxs = sidxs[:sub_n]

        sdata_sample = sdata.select(chosen_idxs)
        all_data.append(sdata_sample)

    all_data = concatenate_datasets(all_data)
    print(f'Saving {len(all_data)} to {args.out_path}...')
    all_data.save_to_disk(args.out_path)
