import os
import regex as re
import string

import argparse
import pandas as pd
import numpy as np
np.random.seed(1992)
import torch
from datasets import load_from_disk, Dataset
from openai import AzureOpenAI
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


PROMPT_TEMPLATE = ("""
<<<
{}
>>>

The previous paragraph demarcated within <<< and >>> comes from {}.

Your job is to assess its educational value to a medical student or resident.

A paragraph with high educational value:
1) Is clearly formatted and non-redundant.
2) Is information dense with medical concepts.
3) Contains specific, non-trivial information about these medical concepts.
4) If about a patient, is necessary to understand the patient's condition or care plan.
5) Can be any length.

Answer with a number from 1 (no educational value) to 5 (high educational value).

DO NOT provide an explanation.
""")


ANSWER_PREFIX = 'Answer (1-5): '


LIKERT_IDS = [
    28740, 28750, 28770, 28781, 28782
]

MODELS = {
    'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'gpt-4': 'gpt-4'
}


SOURCE_DESCRIPTIONS = {
    'pubmed': 'a PubMed Journal Article',
    # 'mimic': 'a Clinical Note',
    'wikidoc': 'a Wikidoc Textbook Chapter',
    'wikipedia': 'a Wikipedia Article',
    'refined_web': 'a Web Page',
    'pes2o': 'an Academic Article',
    'nih_grant_abstracts': 'a Abstract of a Grant submitted to the National Institutes of Health (NIH)',
    'guidelines': 'a Clinical Guideline',
    'gutenberg_books': 'a Book',
    'chemsum': 'an Academic Article on Chemistry',
    'ncbi_bookshelf': 'a Chapter from StatPearls textbook for medical students',
    'medline_plus_genes': 'high quality information from MedLine Plus on a Gene',
    'medline_plus_genetic_conditions': 'high quality information from MedLine Plus on a Genetic Condition',
    'medline_plus_medical_tests': 'high quality information from MedLine Plus on a Medical Test',
    'medline_plus_topic_summaries': 'high quality information from MedLine Plus on a Disease',
}


def gpt_score(prompt, model='gpt-4', temperature=0, max_tokens=2048):
    messages = [
        {'role': 'system', 'content': 'You are a professor at a top medical school and identify highly useful reading material for your students.'},
        {'role': 'user', 'content': prompt}
    ]

    completion = client.with_options(max_retries=5).chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    likert = int(completion.choices[0].message.content)
    assert likert in list(range(1, 6))
    return likert


def mixtral_likert(model, tokenizer, prompt):
    messages = [
        {'role': 'user', 'content': prompt},
    ]

    chat_text = tokenizer.apply_chat_template(messages, tokenize=False) + '\n' + ANSWER_PREFIX
    inputs = tokenizer(chat_text, return_tensors='pt')['input_ids'].to('cuda')
    logits = model(inputs).logits[:, -1, :][0]
    ans_probs = torch.softmax(logits[LIKERT_IDS], dim=0).detach().cpu().numpy()
    expectation = sum([p * x for p, x in zip(ans_probs, range(1, 6))])

    return expectation


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running LLM-based quality filter on paragraphs for different datasets.')

    parser.add_argument('--model', default='mixtral', choices=list(MODELS.keys()))
    parser.add_argument('--paras_per_doc', default=1)
    parser.add_argument('--excluded_sources', default='gutenberg_books|code')
    parser.add_argument('--max_para_toks', default=1024, type=int)
    parser.add_argument('--data_dir', default='/weka/home-griffin/clinical_pile/v1/dataset_hf_50k_sample', type=str)
    parser.add_argument('-overwrite', default=False, action='store_true')

    args = parser.parse_args()

    out_dir = args.data_dir + '_llm_quality_scores'
    os.makedirs(out_dir, exist_ok=True)

    data = load_from_disk('/weka/home-griffin/clinical_pile/v1/dataset_hf_50k_sample')
    N = len(data)

    print('Removing ' + args.excluded_sources)
    excluded_sources = set(args.excluded_sources.split('|'))
    data = data.filter(lambda row: row['source'] not in excluded_sources)
    filt_N = len(data)

    # Shuffle
    idxs = np.arange(filt_N)
    np.random.shuffle(idxs)
    data = data.select(idxs)

    print(f'Left with {filt_N} / {N} examples...')

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

    fns_to_load = []
    for idx, row in tqdm(enumerate(data), total=len(data)):
        out_fn = os.path.join(out_dir, f'{idx}.json')

        fns_to_load.append(out_fn)

        if os.path.exists(out_fn) and not args.overwrite:
            print(f'{out_fn} exists. Run with -overwrite if you want to redo it...')
        else:
            paras = re.split('\n\n', row['text'])
            paras = [p.strip() for p in paras if len(p.strip()) > 0]

            def group_headers(arr):
                new_arr = []
                curr_para = []

                for x in arr:
                    curr_para.append(x)
                    if not x.startswith('#') or '\n' in x:
                        new_arr.append('\n'.join(curr_para))
                        curr_para = []

                if len(curr_para) > 0:
                    new_arr.append('\n'.join(curr_para))
                return new_arr

            # Sometimes headers are alone. If so, group them with next paragraph
            paras = group_headers(paras)

            para_idx = int(np.random.randint(len(paras)))
            para = paras[para_idx]

            if len(para.split(' ')) > args.max_para_toks:
                para = ' '.join(para.split(' ')[:args.max_para_toks])

            source = row['source']

            if source == 'mimic':
                meta = json.loads(row['meta'])
                assert meta['note_type'] in {'Discharge summary', 'Radiology'}
                source_doc_desc = meta['note_type']
                if meta['note_type'] == 'Radiology':
                    source_doc_desc += ' Report'
            else:
                source_doc_desc = SOURCE_DESCRIPTIONS[source]

            # print(para)
            # print('\n')

            prompt = PROMPT_TEMPLATE.format(para, source_doc_desc)

            if args.model == 'mixtral':
                label = mixtral_likert(model, tokenizer, prompt=prompt)
            else:
                label = gpt_score(args.model, prompt=prompt)

            # print(label)
            # print('\n\n\n')
            # print('*' * 50)
            # print('\n\n\n')

            out_row = {
                'uuid': row['uuid'],
                'id': row['id'],
                'source': source,
                'text': para,
                'label': label,
                'paragraph_idx': para_idx,
            }

            with open(out_fn, 'w') as fd:
                json.dump(out_row, fd)

    dataset = Dataset.from_list([
        json.load(open(fn, 'r')) for fn in fns_to_load
    ])

    hf_out = os.path.join(out_dir, 'dataset_hf')
    print(f'Saving {len(dataset)} quality-labeled paragraphs to {hf_out}')
    dataset.save_to_disk(hf_out)
