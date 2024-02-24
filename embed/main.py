import os

import argparse
import json
import h5py
import numpy as np
from datasets import load_dataset
from gritlm import GritLM
import pandas as pd


def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Add embeddings to each document in clinical pile')

    parser.add_argument('--hf_path', default='medarc/clinical_pile_v1_minhash_deduped')
    
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_length', default=2048, type=int)

    parser.add_argument('--chunk', default=None, type=int)
    parser.add_argument('--num_chunks', default=10, type=int)
    parser.add_argument('-single_gpu', default=False, action='store_true')

    parser.add_argument('--save_dir', default='/weka/home-griffin/clinical_pile/v1/embed', type=str)

    args = parser.parse_args()

    if args.single_gpu:
        model = GritLM('GritLM/GritLM-7B', torch_dtype='auto', device_map='auto', mode='embedding')
    else:
        model = GritLM('GritLM/GritLM-7B', torch_dtype='auto', device_map='auto', mode='embedding')

    dataset = load_dataset(args.hf_path, split='train')
    if args.chunk is not None:
        assert args.chunk > 0  # Start at 1 for naming purposes
        all_idxs = np.arange(len(dataset))
        chunk_idxs = np.array_split(all_idxs, args.num_chunks)[args.chunk - 1]
        print(f'Embedding a chunk ({args.chunk} / {args.num_chunks}) of the full dataset from start {chunk_idxs[0]} to end {chunk_idxs[-1]}.')
        dataset = dataset.select(chunk_idxs)
        args.save_dir += f'_{args.chunk}-{args.num_chunks}'

    os.makedirs(args.save_dir, exist_ok=True)

    ids = dataset['id']
    texts = dataset['text']

    embeddings = model.encode(
        texts, instruction=gritlm_instruction('Identify the main topics from a medical document.'),
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    with open(os.path.join(args.save_dir, 'dataset_info.json'), 'w') as fd:
        json.dump({'hf_path': args.hf_path}, fd)

    embed_fn = os.path.join(args.save_dir, 'embeddings.h5')
    print(f'Saving {len(embeddings)} embeddings to {embed_fn}.')
    with h5py.File(embed_fn, 'w') as hf:
        hf.create_dataset('array', data=embeddings)

    meta_dir = os.path.join(args.save_dir, 'dataset_copy_hf')
    dataset = dataset.remove_columns(['text'])
    print(f'Saving non-text columns of dataset to {meta_dir} to match with embeddings.')
    dataset.save_to_disk(meta_dir)
