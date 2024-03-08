from transformers import AutoConfig, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from safetensors import safe_open
import torch
import argparse
import os


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
    parser.add_argument('--experiment', default='baseline_0.5B')
    parser.add_argument('--ckpt', type=int)

    args = parser.parse_args()

    ckpt = os.path.join(args.weight_dir, args.experiment, f'model_state_dict_{args.ckpt}.safetensors')
    out_dir = os.path.join(args.weight_dir, args.experiment, 'hf')

    model = load_model(args.model_name, ckpt=ckpt)
    print(f'Saving model to {out_dir}...')
    model.save_pretrained(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f'Saving tokenizer to {out_dir}...')
    tokenizer.save_pretrained(out_dir)
