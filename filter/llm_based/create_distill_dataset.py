import os
import regex as re
import string

import argparse
import pandas as pd
import numpy as np
np.random.seed(1992)
import torch
from datasets import load_from_disk, DatasetDict


LABELS = [
    'very poor',
    'poor',
    'ok',
    'good',
    'very good'
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Mapping float label to class labels (1-5).')

    parser.add_argument('--val_size', default=1024, type=int)
    parser.add_argument('--data_dir', default='/weka/home-griffin/clinical_pile/v1/dataset_hf_50k_sample_llm_quality_scores/dataset_hf', type=str)
    parser.add_argument('--out_dir', default='/weka/home-griffin/quality_filter/dataset_hf')

    args = parser.parse_args()

    dataset = load_from_disk(args.data_dir)

    dataset = dataset.map(
        lambda row: {'label': LABELS[round(row['label']) - 1]}
    )

    all_idxs = np.arange(len(dataset))
    np.random.shuffle(all_idxs)

    validation_idxs = all_idxs[:args.val_size]
    train_idxs = all_idxs[args.val_size:]

    training_dataset = DatasetDict({
        'train': dataset.select(train_idxs),
        'validation': dataset.select(validation_idxs),
    })

    print(f'Saving to {args.out_dir}')
    training_dataset.save_to_disk(args.out_dir)

    


