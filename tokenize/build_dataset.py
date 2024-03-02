import multiprocess
import argparse
import numpy as np
import os
np.random.seed(1992)
from copy import deepcopy
from itertools import chain
from datasets import load_from_disk
from transformers import AutoTokenizer


from sample_utils import sample_dataset


def should_keep(keep_prob):
    return np.random.random() < keep_prob


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Don't need meta for final tokenized dataset. Just "text" and "source".
    raw_dataset = load_from_disk(args.dataset).remove_columns('meta')

    reweighted_dataset, data_mixture = sample_dataset(raw_dataset, reweighting_config=args.reweighting_config, target_num_tokens=args.target_num_tokens)

    pre_tok_columns = list(reweighted_dataset.features)
    print('Removing all pre-tokenization columns -> ' + ', '.join(pre_tok_columns))

    def tokenize_function(example):
        return {'input_ids': tokenizer([t + tokenizer.eos_token for t in example['text']])['input_ids']}

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

    print(f'Uploading {len(train_tokenized_dataset)} packed tokenized examples from {len(tokenized_dataset)} documents to {args.out_dir}')
    train_tokenized_dataset.save_to_disk(args.out_dir)

    args.out_fn = args.tokenized_dir + '_mixture.csv'

    print(f'Saving information about re-weighted data mixture to {args.out_fn}')
    data_mixture.to_csv(args.out_fn, index=False)

    print('Cleaning up cache files...')
    train_tokenized_dataset.cleanup_cache_files()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Re-weight and tokenize dataset")
    parser.add_argument('--seed', type=int, default=1992, help='Random seed')
    parser.add_argument('--num_proc', default=multiprocess.cpu_count(), type=int)
    parser.add_argument('--max_seq_length', type=int, default=8192, help='Sequence length for processing')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen1.5-0.5B', help='Tokenizer model to use')
    parser.add_argument('--dataset', type=str, default='/weka/home-griffin/clinical_pile/v1/dataset_hf', help='Name of the dataset to process')
    parser.add_argument('--target_num_tokens', type=int, default=int(2e10))
    parser.add_argument('--reweighting_config', type=str, default='all')
    parser.add_argument('--tokenized_dir', default='/weka/home-griffin/clinical_pile/v1/tokenized')
    parser.add_argument('--out_dir', default=None)

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(args.tokenized_dir, f'dataset_hf_{args.reweighting_config}')
        print(f'Did\'nt set --out_dir, so will be pushing dataset to default --> {args.out_dir}')
    
    if os.path.exists(args.out_dir):
        print(f'{args.out_dir} already exists. Remove the file before re-running this script.')
        print(f'rm -rf {args.out_dir}')
        exit(0)

    main(args)
