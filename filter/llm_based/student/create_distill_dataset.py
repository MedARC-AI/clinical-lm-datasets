import os
import regex as re
import string

import argparse
import pandas as pd
import numpy as np
np.random.seed(1992)
from datasets import load_from_disk, DatasetDict


LABELS = [
    'very poor',
    'poor',
    # 'ok',
    'good',
    'very good'
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Mapping float label to class labels (1-4).')

    parser.add_argument('--val_size', default=4096, type=int)
    parser.add_argument('--pile_dir', default='/weka/home-griffin/clinical_pile/v1', type=str)
    parser.add_argument('--out_dir', default='/weka/home-griffin/quality_filter')
    parser.add_argument('--dimension', default='topic', choices=['topic', 'quality'])

    args = parser.parse_args()

    data_dir = os.path.join(args.pile_dir, f'dataset_hf_1mn_sample_llm_{args.dimension}_scores', 'hf')

    print(f'Loading dataset from {data_dir}')
    dataset = load_from_disk(data_dir)

    dataset = dataset.map(
        lambda row: {'label': (row['label'] - 1) / 3.0},  # Normalize 1-4 to 0-1
        num_proc=32
    )

    all_idxs = np.arange(len(dataset))
    np.random.shuffle(all_idxs)

    validation_idxs = all_idxs[:args.val_size]
    train_idxs = all_idxs[args.val_size:]

    training_dataset = DatasetDict({
        'train': dataset.select(train_idxs),
        'validation': dataset.select(validation_idxs),
    })

    out_dir = os.path.join(args.out_dir, args.dimension)
    os.makedirs(out_dir, exist_ok=True)
    print(f'Saving to {out_dir}')
    training_dataset.save_to_disk(out_dir)
