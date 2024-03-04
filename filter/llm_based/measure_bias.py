import os
import regex as re
import string

import argparse
import pandas as pd
import numpy as np
np.random.seed(1992)
import torch
from datasets import load_from_disk
from openai import AzureOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tabulate import tabulate

from gen_labels import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running LLM-based quality filter on paragraphs for different datasets.')

    parser.add_argument('--model', default='mixtral', choices=list(MODELS.keys()))

    args = parser.parse_args()
    
    if args.model == 'mixtral':
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODELS[args.model],
            # torch_dtype='bfloat16',
            load_in_8bit=True,
            attn_implementation='flash_attention_2',
            # quantization_config=quantization_config,
            device_map='auto',
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
        assert tokenizer.decode(LIKERT_IDS) == '12345'
    else:
        assert args.model == 'gpt-4'
        client = AzureOpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
            azure_endpoint=os.environ.get('OPENAI_AZURE_ENDPOINT', None),
            api_version=os.environ.get('OPENAI_API_VERSION', None),
            azure_deployment=os.environ.get('OPENAI_AZURE_DEPLOYMENT', None)
        )

    out_rows = []
    for k, v in SOURCE_DESCRIPTIONS.items():
        out_rows.append({'source': k, 'description': v})

    out_rows.append({'source': 'mimic_dsum', 'description': 'Discharge summary'})
    out_rows.append({'source': 'mimic_report', 'description': 'Radiology Report'})
    
    for row in out_rows:
        prompt = PROMPT_TEMPLATE.format('', row['description'])

        if args.model == 'mixtral':
            label = mixtral_likert(model, tokenizer, prompt=prompt)
        else:
            label = gpt_score(args.model, prompt=prompt)

        row['label'] = label

    out_rows = list(sorted(out_rows, key=lambda x: -x['label']))

    print(tabulate(out_rows))
