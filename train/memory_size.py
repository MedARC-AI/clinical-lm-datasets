import psutil
import transformers
from transformers import AutoModelForCausalLM, AutoConfig
import torch


def print_gpu_memory_usage(device):
    allocated = torch.cuda.memory_allocated(device) / 1e9
    cached = torch.cuda.memory_reserved(device) / 1e9
    print(f"Allocated memory: {allocated:.2f} GB")
    print(f"Cached memory: {cached:.2f} GB")

if __name__ == '__main__':
    hf_model = 'meta-llama/Llama-2-7b-hf' # 'Qwen/Qwen1.5-7B'
    device = 0

    torch.cuda.empty_cache() # Clear any cached memory

    print_gpu_memory_usage(device)

    # Load your model (example)
    # model = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype=torch.bfloat16).to(device)

    print(f'Loading config from {hf_model}')
    cfg = AutoConfig.from_pretrained(hf_model)
    cfg.use_cache = False
    cfg._attn_implementation = 'spda'
    # with init_empty_weights():
    print('Loading model using config')
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16).to(device)
    print('Loaded Model...')

    print_gpu_memory_usage(device)
    
    model.cpu()
