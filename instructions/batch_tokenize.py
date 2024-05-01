import multiprocess
import argparse
import numpy as np
np.random.seed(1992)
from itertools import chain
from datasets import load_from_disk
from transformers import AutoTokenizer


TOKENIZERS = {
    'qwen': 'Qwen/Qwen1.5-0.5B',
    'llama2': 'meta-llama/Llama-2-7b-hf',
    'stable': 'stabilityai/stablelm-3b-4e1t',
    'llama3': 'meta-llama/Meta-Llama-3-8B',
}


def combine(row):
    prompt = row['prompt']
    completion = row['completion']
    
    if prompt.endswith('# ANSWER\n'):
        return prompt + completion
    if '# ANSWER' not in completion:
        print(row)
        raise Exception("OUT")
    return prompt.strip() + '\n\n' + completion.strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Re-weight and tokenize dataset")
    parser.add_argument('--num_proc', default=multiprocess.cpu_count() - 8, type=int)
    parser.add_argument('--model', type=str, default='llama3', help='Tokenizer model to use', choices=TOKENIZERS.keys())
    parser.add_argument('--dataset', type=str, default='/weka/home-griffin/clinical_instructions/v1/dataset_hf', help='Name of the dataset to process')

    args = parser.parse_args()

    out_dir = args.dataset + f'_{args.model}_tokenized'
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZERS[args.model])

    def tokenize_function(example):
        return {'input_ids': tokenizer(combine(example) + tokenizer.eos_token)['input_ids']}
    
    # Don't need meta for tokenized dataset. Just "text" and "source".
    dataset = load_from_disk(args.dataset)

    print('Tokenizing and removing text...')

    tokenized_dataset = dataset.map(
        tokenize_function,
        num_proc=args.num_proc,
        remove_columns=['prompt', 'completion'],
    )

    print(f'Saving to {out_dir}...')
    tokenized_dataset.save_to_disk(out_dir)
