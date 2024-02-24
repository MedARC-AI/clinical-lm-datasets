import os

import argparse
import h5py
from datasets import load_dataset
from gritlm import GritLM
import pandas as pd


def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Add embeddings to each document in clinical pile')

    args = parser.parse_args()

    parser.add_argument('--hf_path', default='medarc/clinical-pile-v1')
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_length', default=4096, type=int)

    parser.add_argument('--save_dir', default='/weka/home-griffin/clinical_pile/v1/embed', type=str)

    args = parser.parse_args()

    model = GritLM('GritLM/GritLM-7B', torch_dtype='auto', device_map='auto', mode='embedding')

    dataset = load_dataset(args.hf_path, split='train')
    ids = dataset['id']
    texts = dataset['text']

    embeddings = model.encode(
        texts, instruction=gritlm_instruction(''),
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    embed_fn = os.path.join(args.save_dir, 'embeddings.h5')
    print(f'Saving {len(embeddings)} embeddings to {embed_fn}.')
    with h5py.File(embed_fn, 'w') as hf:
        hf.create_dataset('array', data=embeddings)

    meta_dir = os.path.join(args.save_dir, 'dataset_copy_hf')
    dataset = dataset.remove_columns(['text'])
    print(f'Saving non-text columns of dataset to {meta_dir} to match with embeddings.')
    dataset.save_to_disk(meta_dir)
