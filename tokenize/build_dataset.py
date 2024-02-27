import multiprocessing
import argparse
from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train_dataset = load_dataset(args.dataset, split='train')

    pre_tok_columns = list(train_dataset.features)
    print('Removing all pre-tokenization columns -> ' + ', '.join(pre_tok_columns))

    def tokenize_function(example):
        return {'input_ids': tokenizer([t + tokenizer.eos_token for t in example['text']])['input_ids']}

    tokenized_dataset = train_dataset.map(
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
        return result

    train_tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=int(1e-6),
        num_proc=args.num_proc
    )

    out_dir = args.dataset + '_tokenized'
    print(f'Uploading {len(train_tokenized_dataset)} packed tokenized examples from {len(tokenized_dataset)} documents to {out_dir}')
    train_tokenized_dataset.push_to_hub(out_dir, private=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and push dataset to Hugging Face Hub")
    parser.add_argument('--seed', type=int, default=1992, help='Random seed')
    parser.add_argument('--num_proc', default=multiprocessing.cpu_count(), type=int)
    parser.add_argument('--max_seq_length', type=int, default=8192, help='Sequence length for processing')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen1.5-72B', help='Tokenizer model to use')
    parser.add_argument('--dataset', type=str, default='medarc/clinical_pile_v1_minhash_deduped', help='Name of the dataset to process')
    
    args = parser.parse_args()
    
    main(args)
