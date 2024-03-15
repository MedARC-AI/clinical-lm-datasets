import multiprocess
import argparse
import json
import numpy as np
import os
np.random.seed(1992)
from itertools import chain
import datasets
from datasets import load_from_disk
import h5py
from tqdm import tqdm

from sample_utils import sample_dataset
from batch_tokenize import TOKENIZERS


def pack(args):
    data_dir = os.path.join(args.pile_dir, f'dataset_hf_clean_{args.model}_tokenized')

    # Don't need meta for final tokenized, packed dataset. Just "input_ids" and "source".
    raw_dataset = load_from_disk(data_dir).remove_columns('meta')

    reweighted_dataset, data_mixture = sample_dataset(raw_dataset, reweighting_config=args.reweighting_config, target_num_tokens=args.target_num_tokens)

    # Remove any column != input_ids
    remove_cols = [col for col in reweighted_dataset.features if col != 'input_ids']
    reweighted_dataset = reweighted_dataset.remove_columns(remove_cols)

    block_size = args.max_seq_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        return result

    reweighted_dataset = reweighted_dataset.map(
        group_texts,
        batched=True,
        num_proc=args.num_proc
    )

    shape = (len(reweighted_dataset), args.max_seq_length)  # ~ 10-20 million rows, max_seq_length columns
    dtype = np.int32  # Assuming 32-bit integers
    fp = np.memmap(args.out_fn, dtype=dtype, mode='w+', shape=shape)

    # Iterate over the dataset and store the data in the memory-mapped array
    for i, item in tqdm(enumerate(reweighted_dataset), total=len(reweighted_dataset)):
        fp[i] = item['input_ids']

    stats_fn = args.out_fn.split('.')[0] + '_mixture.csv'

    print(f'Saving information about re-weighted data mixture to {stats_fn}')
    data_mixture.to_csv(stats_fn, index=False)

    print('Cleaning up cache files...')
    reweighted_dataset.cleanup_cache_files()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Re-weight and tokenize dataset")
    parser.add_argument('--seed', type=int, default=1992, help='Random seed')
    parser.add_argument('--num_proc', default=multiprocess.cpu_count() - 16, type=int)
    parser.add_argument('--max_seq_length', type=int, default=8192, help='Sequence length for processing')
    parser.add_argument('--pile_dir', type=str, default='/weka/home-griffin/clinical_pile/v1', help='Name of the dataset to process')
    parser.add_argument('--target_num_tokens', type=int, default=int(1e11)) # 100 billion
    parser.add_argument('--reweighting_config', type=str, default='all')
    parser.add_argument('-all_configs', default=False, action='store_true')
    parser.add_argument('--out_dir', default='/weka/home-griffin/clinical_pile/v1/packed')
    parser.add_argument('--model', type=str, default='llama2')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.all_configs:
        fns = os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mixtures'))
        reweighting_configs = []
        for fn in fns:
            config = fn.split('/')[-1].replace('.csv', '')
            reweighting_configs.append(config)

        reweighting_configs = list(sorted(reweighting_configs))
        print('Pre-processing all configs with -all_configs file, as found in ./mixtures')
        print(reweighting_configs)
        for config in reweighting_configs:
            args.reweighting_config = config
            args.out_fn = os.path.join(args.out_dir, f'{args.reweighting_config}_{args.model}_{args.max_seq_length}.memmap')
            pack(args)
    else:
        args.out_fn = os.path.join(args.out_dir, f'{args.reweighting_config}_{args.model}_{args.max_seq_length}.memmap')
        pack(args)
    