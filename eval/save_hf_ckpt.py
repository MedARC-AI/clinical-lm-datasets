from collections import defaultdict, Counter
import torch
import argparse
import os
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
from ckpt_to_hf import load_model
import regex as re

from glob import glob
import numpy as np
np.random.seed(4692)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save a safetensor and tokenizer as a HF checkpoint for easy model loading with from_pretrained.')
    
    parser.add_argument('--model_name', default='Qwen/Qwen1.5-7B')
    
    parser.add_argument('--weight_dir', default='/weka/home-griffin/weights/finetune/Qwen/Qwen1.5-7B')
    parser.add_argument('--experiment', default='qwen_7b_base_pubmedqa_v2')
    parser.add_argument('--ckpt', type=int, default=-1)  # -1 means take the last checkpoint

    args = parser.parse_args()

    if args.ckpt == -1:
        print(f'Searching for latest checkpoint in {args.ckpt}...')
        pattern = os.path.join(args.weight_dir, args.experiment, '*.safetensors')
        all_ckpts = list(glob(pattern))

        if len(all_ckpts) == 0:
            print(f'No checkpoints found in {os.path.join(args.weight_dir, args.experiment)}.')
        args.ckpt = max([int(re.search('model_state_dict_(\d+)', fn).group(1)) for fn in all_ckpts])
        print(f'Found checkpoint --> {args.ckpt}')
        ckpt_name = 'final'
        model_dir = os.path.join(args.weight_dir, args.experiment, f'hf_{ckpt_name}')
    else:
        ckpt_name = str(args.ckpt)
        model_dir = os.path.join(args.weight_dir, args.experiment, f'hf_{ckpt_name}')

    if not os.path.exists(model_dir):
        print(f'Loading tokenizer from {args.model_name}')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        ckpt = os.path.join(args.weight_dir, args.experiment, f'model_state_dict_{args.ckpt}.safetensors')
        print(f'Converting {ckpt} to HF dataset...')
        model = load_model(args.model_name, ckpt=ckpt)
        print(f'Saving model and tokenizer to {model_dir}...')
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        print(f'{model_dir} already exists!')