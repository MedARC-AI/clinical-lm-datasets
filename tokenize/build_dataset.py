import multiprocess
import argparse
import json
import numpy as np
import os
np.random.seed(1992)
from copy import deepcopy
from itertools import chain
import datasets
from datasets import load_from_disk, get_dataset_infos
from transformers import AutoTokenizer
import h5py
from tqdm import tqdm
from p_tqdm import p_uimap

from sample_utils import sample_dataset


def should_keep(keep_prob):
    return np.random.random() < keep_prob


def dump_jsonl(args):
    # Don't need meta for final tokenized dataset. Just "text" and "source".
    raw_dataset = load_from_disk(args.dataset).remove_columns('meta')

    reweighted_dataset, data_mixture = sample_dataset(raw_dataset, reweighting_config=args.reweighting_config, target_num_tokens=args.target_num_tokens)

    print(f'Writing {len(reweighted_dataset)} lines to {args.out_fn}')
    with open(args.out_fn, 'w') as f:
        for row in tqdm(reweighted_dataset):
            f.write(json.dumps({'text': row['text']}))
            f.write('\n')


def push_to_hub(args, hub_dir):
    # Don't need meta for final tokenized dataset. Just "text" and "source".
    raw_dataset = load_from_disk(args.dataset).remove_columns('meta')

    reweighted_dataset, data_mixture = sample_dataset(raw_dataset, reweighting_config=args.reweighting_config, target_num_tokens=args.target_num_tokens)

    reweighted_dataset.push_to_hub(hub_dir, private=True)


def tokenize(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Don't need meta for final tokenized dataset. Just "text" and "source".
    raw_dataset = load_from_disk(args.dataset).remove_columns('meta')

    reweighted_dataset, data_mixture = sample_dataset(raw_dataset, reweighting_config=args.reweighting_config, target_num_tokens=args.target_num_tokens)

    def tokenize_function(example):
        return {'input_ids': tokenizer([t + tokenizer.eos_token for t in example['text']])['input_ids']}

    pre_tok_columns = list(reweighted_dataset.features)
    print('Tokenizing and removing all pre-tokenization columns -> ' + ', '.join(pre_tok_columns))
    
    tokenized_dataset = reweighted_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=pre_tok_columns,
    )

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

        # Add in attention mask == all ones
        # Add in labels == input_ids
        # result['labels'] = deepcopy(result['input_ids'])
        # result['attention_mask'] = [[1 for _ in range(len(arr))] for arr in result['input_ids']]

        return result

    train_tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=args.num_proc
    )

    os.makedirs(args.out_dir, exist_ok=True)

    def process_shard(shard_idx, shard, show_progress=False):
        out_fn = os.path.join(args.out_dir, f'{shard_idx}.h5')
        print(f'Saving shard {shard_idx} / {args.shards} to {out_fn}...')
        with h5py.File(out_fn, 'w') as f:
            if show_progress:  # Slower but shows progress and doesn't crash / hang.
                out_data = f.create_dataset('input_ids', shape=(len(shard), 8192), maxshape=(len(shard), 8192))
                from tqdm import tqdm
                out_idx = 0
                for row in tqdm(shard):
                    out_data[out_idx] = row['input_ids']
                    out_idx += 1
            else: # Fastest but sometimes hangs / crashes and doesn't tell you why
                f.create_dataset('input_ids', data=shard['input_ids'])
        print(f'DONE! Saved shard {shard_idx} / {args.shards} to {out_fn}!')

    print('Dumping each shard to h5 file separately...')
    list(p_uimap(lambda shard_idx: process_shard(shard_idx, train_tokenized_dataset.shard(num_shards=args.shards, index=shard_idx, contiguous=True)), range(args.shards), num_cpus=args.shards))

    # list(tqdm(map(lambda shard_idx: process_shard(shard_idx, train_tokenized_dataset.shard(num_shards=args.shards, index=shard_idx, contiguous=True)), range(args.shards))))

    # print(f'Uploading {len(train_tokenized_dataset)} packed tokenized examples from {len(tokenized_dataset)} documents to {args.out_dir}')
    # train_tokenized_dataset.save_to_disk(args.out_dir)

    args.out_fn = args.tokenized_dir + '_mixture.csv'

    print(f'Saving information about re-weighted data mixture to {args.out_fn}')
    data_mixture.to_csv(args.out_fn, index=False)

    print('Cleaning up cache files...')
    train_tokenized_dataset.cleanup_cache_files()


def main(args):
    if not args.axolotl and os.path.exists(args.out_dir):
        print(f'{args.out_dir} already exists. Remove the file before re-running this script.')
        print(f'rm -rf {args.out_dir}')
        return

    # if args.out_fn is not None and os.path.exists(args.out_fn):
    #     print(f'{args.out_fn} already exists. Remove the file before re-running this script.')
    #     print(f'rm {args.out_fn}')
    #     exit(0)

    if args.axolotl:
        hub_dir = f'medarc/clinical_pile_v1_{args.reweighting_config}'

        with open(args.axolotl_config_template) as fd:
            axolotl_config = fd.read().strip().replace('{{CONFIG}}', args.reweighting_config)
            axolotl_config_out_fn = args.axolotl_config_template.replace('.yml', '') + f'_{args.reweighting_config}.yml'
            print(f'Saving axolotl config for {args.reweighting_config} to {axolotl_config_out_fn}')
            with open(axolotl_config_out_fn, 'w') as fd:
                fd.write(axolotl_config)

        try:
            existing_info = get_dataset_infos(hub_dir)
            print(f'{hub_dir} already exists on the Hub. Remove it before re-running this script.')
            print(existing_info)
        except datasets.exceptions.DatasetNotFoundError:
            push_to_hub(args, hub_dir=hub_dir)
    else:
        tokenize(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Re-weight and tokenize dataset")
    parser.add_argument('--seed', type=int, default=1992, help='Random seed')
    parser.add_argument('--num_proc', default=multiprocess.cpu_count(), type=int)
    parser.add_argument('--max_seq_length', type=int, default=8192, help='Sequence length for processing')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen1.5-0.5B', help='Tokenizer model to use')
    parser.add_argument('--dataset', type=str, default='/weka/home-griffin/clinical_pile/v1/dataset_hf', help='Name of the dataset to process')
    parser.add_argument('--target_num_tokens', type=int, default=int(1e10))
    parser.add_argument('--reweighting_config', type=str, default='all')
    parser.add_argument('-all_configs', default=False, action='store_true')
    parser.add_argument('--tokenized_dir', default='/weka/home-griffin/clinical_pile/v1')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--out_fn', default=None)
    parser.add_argument('-axolotl', default=False, action='store_true')
    parser.add_argument('--axolotl_config_template', default='/weka/home-griffin/axolotl/ablations.yml')

    parser.add_argument('--shards', default=8, type=int)

    args = parser.parse_args()

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
            args.out_dir = os.path.join(args.tokenized_dir, 'tokenized', f'dataset_hf_{args.reweighting_config}')
            main(args)
    else:
        if not args.axolotl and args.out_dir is None:
            # if args.axolotl:
            #     args.out_dir = os.path.join(args.tokenized_dir, 'axolotl')
            #     os.makedirs(args.out_dir, exist_ok=True)
            #     args.out_fn = os.path.join(args.out_dir, f'{args.reweighting_config}.jsonl')
            #     print(f'Didn\'t set --out_fn, so will be pushing dataset to default --> {args.out_fn}')
            # else:
            args.out_dir = os.path.join(args.tokenized_dir, 'tokenized', f'dataset_hf_{args.reweighting_config}')
            print(f'Didn\'t set --out_dir, so will be pushing dataset to default --> {args.out_dir}')

        main(args)