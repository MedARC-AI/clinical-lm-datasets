import os

import argparse
import json
import h5py
import numpy as np
from datasets import load_from_disk
from gritlm import GritLM
import numpy as np
from tqdm import tqdm


def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Add embeddings to each document in clinical pile')

    parser.add_argument('--data_path', default='/weka/home-griffin/clinical_pile/v1/dataset_hf_clean')
    
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_length', default=2048, type=int)

    parser.add_argument('--chunk', default=None, type=int)
    parser.add_argument('--num_chunks', default=10, type=int)
    parser.add_argument('--num_shards', default=100, type=int)

    parser.add_argument('--save_dir', default='/weka/home-griffin/clinical_pile/v1/embeddings', type=str)

    args = parser.parse_args()

    model = GritLM('GritLM/GritLM-7B', torch_dtype='auto', device_map='auto', mode='embedding')

    # dataset = load_dataset(args.hf_path, split='train')
    dataset = load_from_disk(args.data_path)
    if args.chunk is not None:
        assert args.chunk > 0  # Start at 1 for naming purposes
        all_idxs = np.arange(len(dataset))
        chunk_idxs = np.array_split(all_idxs, args.num_chunks)[args.chunk - 1]
        print(f'Embedding a chunk ({args.chunk} / {args.num_chunks}) of the full dataset from start {chunk_idxs[0]} to end {chunk_idxs[-1]}.')
        dataset = dataset.select(chunk_idxs)
        args.save_dir = os.path.join(args.save_dir, f'{args.chunk}-{args.num_chunks}')

    os.makedirs(args.save_dir, exist_ok=True)

    for shard in tqdm(range(args.num_shards)):
        print(f'Processing Shard={shard}/{args.num_shards} for Chunk={args.chunk}/{args.num_chunks}')

        shard_dir = os.path.join(args.save_dir, f'{shard}-{args.num_shards}')
        embed_fn = os.path.join(shard_dir, 'embeddings.h5')
        id_dir = os.path.join(shard_dir, 'ids')

        if os.path.exists(embed_fn):
            print(f'{embed_fn} exists. Skipping...')
            continue

        shard_hf = dataset.shard(num_shards=args.num_shards, index=shard)

        embeddings = model.encode(
            shard_hf['text'], instruction=gritlm_instruction('Identify the main topics from a medical document.'),
            batch_size=args.batch_size,
            max_length=args.max_length
        )

        os.makedirs(shard_dir, exist_ok=True)
        print(f'Saving {len(embeddings)} embeddings to {embed_fn}.')
        with h5py.File(embed_fn, 'w') as hf:
            hf.create_dataset('array', data=embeddings)

        remove_cols = [x for x in shard_hf.column_names if x not in {'id', 'uuid', 'source'}]
        print(f'Saving non-text columns of dataset to {id_dir} to match with embeddings.')
        shard_hf.remove_columns(remove_cols).save_to_disk(id_dir)

    with open(os.path.join(args.save_dir, 'dataset_info.json'), 'w') as fd:
        json.dump({'data_path': args.data_path}, fd)
