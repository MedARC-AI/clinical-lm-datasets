import os
import shutil
import pandas as pd
from datasets import Dataset, load_from_disk
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import regex as re
from transformers import pipeline
import torch
import multiprocess
from scipy.stats import describe
import argparse
import numpy as np
np.random.seed(1992)
from filter.llm_based.teacher.gen_labels import group_headers


PRECOMPUTED_AVG_SCORES = {
    'longformer_quality': 0.6576033734135329,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Filter')

    parser.add_argument('--weight_dir', default='/weka/home-griffin/weights/quality-filter')
    parser.add_argument('--dimension', default='quality')
    parser.add_argument('--experiment', default='longformer_quality')
    parser.add_argument('--pile_dir', default='/weka/home-griffin/clinical_pile/v1/dataset_hf_clean')
    parser.add_argument('--out_dir', default='/weka/home-griffin/clinical_pile/v1/ask_llm_shards')
    parser.add_argument('--min_para_toks', default=32, type=int)
    parser.add_argument('--target_keep_percent', default=0.65, type=float)
    parser.add_argument('--excluded_sources', default='code|gutenberg_books')

    parser.add_argument('--shard_idx', default=0, type=int)
    parser.add_argument('--num_shards', default=100, type=int)
    
    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument('-compute_avg_pred', default=False, action='store_true')
    parser.add_argument('-overwrite', default=False, action='store_true')
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)

    shard_dir = os.path.join(args.out_dir, f'{args.shard_idx}_{args.num_shards}')

    if os.path.exists(shard_dir) and not args.overwrite:
        print(f'{shard_dir} already exists. To overwrite, run with -overwrite or manually "rm -rf {shard_dir}"')
        exit(0)
    elif os.path.exists(shard_dir):
        assert args.overwrite
        print(f'{shard_dir} already exists and you ran with -overwrite. Deleting first...')
        shutil.rmtree(shard_dir)

    model_path = os.path.join(args.weight_dir, args.dimension, args.experiment)

    assert os.path.exists(model_path)

    pipe = pipeline('text-classification', model=model_path, device='cuda', torch_dtype=torch.bfloat16)

    data = load_from_disk(args.pile_dir)

    if args.compute_avg_pred:
        data = data.shuffle(seed=1992).select(list(range(10000)))
        para_dataset = []
        for row in data:
            text = row['text']
            paras = re.split('\n\n', text)
            paras = [p.strip() for p in paras if len(p.strip()) > 0]
            # Sometimes headers are alone. If so, group them with next paragraph
            paras = group_headers(paras, min_para_toks=args.min_para_toks)
            para = np.random.choice(paras)
            para_dataset.append({
                'text': para
            })

        para_dataset = Dataset.from_list(para_dataset)
        scores = []
        for out in tqdm(pipe(KeyDataset(para_dataset, 'text'), truncation=True, batch_size=args.batch_size), total=len(para_dataset)):
            scores.append(out['score'])

        print(describe(scores))

        exit(0)

    print(f'Taking shard {args.shard_idx} / {args.num_shards}')
    data = data.shard(num_shards=args.num_shards, index=args.shard_idx, contiguous=True)

    # Filter out excluded sources
    excluded_sources = set(args.excluded_sources.split('|'))
    data = data.filter(
        lambda row: row['source'] not in excluded_sources, num_proc=multiprocess.cpu_count() - 8
    )        

    if len(data) == 0:
        print('All data in this shard is excluded. Saving a blank directory so it is marked as done.')
        os.makedirs(shard_dir, exist_ok=True)
        exit(0)

    para_dataset = []

    args.sampling_adj = args.target_keep_percent - PRECOMPUTED_AVG_SCORES[args.experiment]
    print(f'Will be adding {args.sampling_adj} to each of the raw probablities so that we keep {args.target_keep_percent} of the original tokens.')

    for row in tqdm(data):
        text = row['text']
        paras = re.split('\n\n', text)
        paras = [p.strip() for p in paras if len(p.strip()) > 0]
        # Sometimes headers are alone. If so, group them with next paragraph
        paras = group_headers(paras, min_para_toks=args.min_para_toks)

        # Run through inference
        uuid = row['uuid']
        source = row['source']

        for para_idx, para in enumerate(paras):
            para_dataset.append({
                'uuid': uuid,
                'para_idx': para_idx,
                'text': para,
                'source': source,
            })

    para_dataset = Dataset.from_list(para_dataset)

    scores = []
    for out in tqdm(pipe(KeyDataset(para_dataset, 'text'), truncation=True, batch_size=50), total=len(para_dataset)):
        scores.append(out['score'])

    para_dataset = para_dataset.add_column('score', scores)

    # We should have at least 1 paragraph for each document (unique UUID)
    assert len(set(para_dataset['uuid'])) == len(data)

    filt_texts = []
    filt_scores = []
    prev_uuid = None
    curr_paras = []
    curr_scores = []

    from collections import defaultdict, Counter
    kept_by_source = defaultdict(int)
    num_by_source = Counter(para_dataset['source'])
    kept = 0
    for row in tqdm(para_dataset):
        uuid = row['uuid']

        if prev_uuid is not None and prev_uuid != uuid:
            # Dump what we have
            filt_texts.append('\n\n'.join(curr_paras))
            filt_scores.append(0 if len(curr_scores) == 0 else sum(curr_scores) / len(curr_scores))
            curr_paras = []
            curr_scores = []

        # Change it to sampling
        adjusted_score = row['score'] + args.sampling_adj
        if adjusted_score <= 0:
            # No need to compute random number
            pass
        elif adjusted_score >= 1 or adjusted_score >= np.random.random():
            curr_scores.append(row['score'])
            curr_paras.append(row['text'])
            kept += 1
            kept_by_source[row['source']] += 1

        prev_uuid = uuid

    print(f'Kept {kept} / {len(para_dataset)} overall paragraphs.')

    for source, full_num in num_by_source.items():
        kept_num = kept_by_source[source]
        print(f'Kept {kept_num} / {full_num} ({round(kept_num / full_num * 100, 1)}%) {source} paragraphs')

    filt_texts.append('\n\n'.join(curr_paras))
    filt_scores.append(0 if len(curr_scores) == 0 else sum(curr_scores) / len(curr_scores))

    assert len(data) == len(filt_scores) == len(filt_texts)

    # Replace old "text" with filt_texts
    data = data.remove_columns('text')
    data = data.add_column('text', filt_texts)
    data = data.add_column('score', filt_scores)

    print(f'Saving to {shard_dir}...')
    data.save_to_disk(shard_dir)
