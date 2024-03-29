from transformers import AutoConfig, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from safetensors import safe_open
import torch
from glob import glob
import argparse
import os
import regex as re


def load_model(model_name, ckpt):
    config = AutoConfig.from_pretrained(model_name)
    config._attn_implementation = 'sdpa'
    config.torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_config(
        config,
    )

    weights = {}
    with safe_open(ckpt, framework='pt', device='cpu') as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    
    print(f'Loading state dict with {len(weights)} keys')
    status = model.load_state_dict(weights, strict=True)
    print(status)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert a SafeTensors checkpoint into a HuggingFace dataset so it can be loaded with "from_pretrained".')
    
    parser.add_argument('--model_name', default='Qwen/Qwen1.5-0.5B')
    
    parser.add_argument('--weight_dir', default='/weka/home-griffin/weights/pretrain/Qwen/Qwen1.5-0.5B')
    parser.add_argument('--experiment', default='pubmed_v2')
    parser.add_argument('--ckpt', type=int, default=-1)

    args = parser.parse_args()

    if args.ckpt == -1:
        out_dir = os.path.join(args.weight_dir, args.experiment, 'hf_final')
        
        print(f'Searching for latest checkpoint in {args.ckpt}...')
        pattern = os.path.join(args.weight_dir, args.experiment, '*.safetensors')
        all_ckpts = list(glob(pattern))

        if len(all_ckpts) == 0:
            print(f'No checkpoints found in {os.path.join(args.weight_dir, args.experiment)}.')
        args.ckpt = max([int(re.search('model_state_dict_(\d+)', fn).group(1)) for fn in all_ckpts])
        print(f'Found checkpoint --> {args.ckpt}')
    else:
        out_dir = os.path.join(args.weight_dir, args.experiment, f'hf_{args.ckpt}')
    
    ckpt_path = os.path.join(args.weight_dir, args.experiment, f'model_state_dict_{args.ckpt}.safetensors')

    if not os.path.exists(ckpt_path):
        print(f'Could not find Checkpoint --> {ckpt_path}. But we do have...')
        for dir in os.listdir(os.path.join(args.weight_dir, args.experiment)):
            print(dir)
        exit(0)

    model = load_model(args.model_name, ckpt=ckpt_path)
    print(f'Saving model to {out_dir}...')
    model.save_pretrained(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f'Saving tokenizer to {out_dir}...')
    tokenizer.save_pretrained(out_dir)
