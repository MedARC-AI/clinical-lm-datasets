import multiprocess
import argparse
import numpy as np
np.random.seed(1992)
from itertools import chain
from datasets import load_from_disk
from transformers import AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Re-weight and tokenize dataset")
    parser.add_argument('--num_proc', default=multiprocess.cpu_count() - 16, type=int)
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen1.5-0.5B', help='Tokenizer model to use')
    parser.add_argument('--dataset', type=str, default='/weka/home-griffin/clinical_pile/v1/dataset_hf', help='Name of the dataset to process')

    args = parser.parse_args()

    out_dir = args.dataset + '_tokenized'
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def tokenize_function(example):
        return {'input_ids': tokenizer([t + tokenizer.eos_token for t in example['text']])['input_ids']}
    
    # Don't need meta for tokenized dataset. Just "text" and "source".
    dataset = load_from_disk(args.dataset)

    print('Tokenizing and removing text...')

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=['text'],
    )

    print(f'Saving to {out_dir}...')
    tokenized_dataset.save_to_disk(out_dir)
