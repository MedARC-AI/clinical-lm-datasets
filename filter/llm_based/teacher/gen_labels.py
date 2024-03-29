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


MEDICAL_TEMPLATE = ("""
<<<
{}
>>>

Does the previous paragraph demarcated within <<< and >>> contain relevant content for training a biomedical language model (LLM)?
- Relevant topics in the field of biomedicine include, but are not limited to, medical research findings, clinical procedures and practices, healthcare policies, patient care strategies, pharmaceutical developments, or biotechnological innovations.

Answer the question with a number from 1 (Strongly Disagree) to 4 (Strongly Agree):
1) Strongly Disagree: The paragraph does not contain information pertinent to biomedicine or healthcare.
2) Disagree: The paragraph mentions biomedical topics only in passing or as minor, uninformative details within a broader context not directly related to biomedicine.
3) Agree: The paragraph contains elements related to biomedicine but is not primarily focused on the field. This might include general science articles with snippets pertinent to medical applications, medical text with mostly references or boilerplate, or discussions on policy or ethics with implications for healthcare.
4) Strongly Agree: The paragraph exclusively focuses on biomedical sciences, clinical research or practice, medical technologies, healthcare policies, or patient care strategies.

DO NOT provide an explanation.
""")


QUALITY_TEMPLATE = ("""
<<<
{}
>>>

Does the previous paragraph demarcated within <<< and >>> contain a highly informative signal for pre-training a Large Language Model (LLM)?
- An informative datapoint is highly educational and contains useful, specific, and non-trivial knowledge about the world.

Answer the question with a number from 1 (Strongly Disagree) to 4 (Strongly Agree):
1) Strongly Disagree: The paragraph has little to no educational value.
2) Disagree: The paragraph has minor educational value.
3) Agree: The paragraph has moderate educational value.
4) Strongly Agree: The paragraph has high educational value.

DO NOT provide an explanation.
""")


TEMPLATES = {
    'quality': QUALITY_TEMPLATE,
    'topic': MEDICAL_TEMPLATE,
}


ANSWER_PREFIX = 'Answer (1-4): '

# Hardcode just to be extra sure
LIKERT_IDS = [
    28740, 28750, 28770, 28781  # , 28782
]

MODELS = {
    'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'gpt-4': 'gpt-4'
}

def group_headers(arr, min_para_toks):
    new_arr = []
    curr_para = []

    for x in arr:
        curr_para.append(x)

        num_curr_toks = len('\n'.join(curr_para).split(' '))

        if x.startswith('#') and '\n' not in x:  # Likely a header
            end_para = False
        elif num_curr_toks < min_para_toks:
            end_para = False
        else:
            end_para = True
        
        if end_para:
            new_arr.append('\n'.join(curr_para))
            curr_para = []

    if len(curr_para) > 0:
        new_arr.append('\n'.join(curr_para))
    return new_arr


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
    assert likert in list(range(1, 5))
    return likert


def mixtral_likert(model, tokenizer, prompt):
    messages = [
        {'role': 'user', 'content': prompt},
    ]

    chat_text = tokenizer.apply_chat_template(messages, tokenize=False) + '\n\n' + ANSWER_PREFIX
    inputs = tokenizer(chat_text, return_tensors='pt')['input_ids'].to('cuda')

    assert inputs.dim() == 2
    # Eventually remove
    print(inputs.size()[1])
    if inputs.size()[1] >= 2048:
        # Halve the prompt
        return mixtral_likert(model, tokenizer, prompt[:len(prompt) // 2])

    logits = model(inputs).logits[:, -1, :][0]
    ans_probs = torch.softmax(logits[LIKERT_IDS], dim=0).detach().cpu().numpy()
    expectation = sum([p * x for p, x in zip(ans_probs, range(1, 6))])

    return expectation


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running LLM-based quality filter on paragraphs for different datasets.')

    parser.add_argument('--model', default='mixtral', choices=list(MODELS.keys()))
    parser.add_argument('--excluded_sources', default=None)
    parser.add_argument('--max_para_toks', default=512, type=int)
    parser.add_argument('--min_para_toks', default=32, type=int)
    parser.add_argument('--dimension', default='quality', choices=['quality', 'topic'])
    parser.add_argument('--data_dir', default='/weka/home-griffin/clinical_pile/v1/dataset_hf_1mn_sample', type=str)
    parser.add_argument('-overwrite', default=False, action='store_true')

    parser.add_argument('--chunk', default=None, type=int)
    parser.add_argument('--num_chunks', default=10, type=int)

    args = parser.parse_args()

    out_dir = args.data_dir + f'_llm_{args.dimension}_scores'
    os.makedirs(out_dir, exist_ok=True)

    data = load_from_disk(args.data_dir)
    # Original Dataset Index for saving the file
    data = data.map(
        lambda row, idx: {'dataset_idx': idx},
        with_indices=True
    )
    N = len(data)

    if args.chunk is not None:
        data = data.shard(num_shards=args.num_chunks, index=args.chunk, contiguous=True)

    if args.excluded_sources is not None:
        print('Removing ' + args.excluded_sources)
        excluded_sources = set(args.excluded_sources.split('|'))
        data = data.filter(lambda row: row['source'] not in excluded_sources)
        filt_N = len(data)
        print(f'Left with {filt_N} / {N} examples...')

    cache_dir = os.path.join(out_dir, 'hf')
    if os.path.exists(cache_dir):
        cached_labels = load_from_disk(cache_dir)
        done_uuids = set(cached_labels['uuid'])
        print(f'Loaded {len(done_uuids)} already generated labels from {cache_dir}. Filtering them out of shard.')
        data = data.filter(
            lambda row: row['uuid'] not in done_uuids, num_proc=32
        )

    if args.model == 'mixtral':
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )

        print(f'Loading {MODELS[args.model]}...')
        model = AutoModelForCausalLM.from_pretrained(
            MODELS[args.model],
            # torch_dtype='bfloat16',
            load_in_8bit=True,
            attn_implementation='flash_attention_2',
            # quantization_config=quantization_config,
            device_map='auto',
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
        assert tokenizer.decode(LIKERT_IDS) == '1234'
    else:
        assert args.model == 'gpt-4'
        client = AzureOpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
            azure_endpoint=os.environ.get('OPENAI_AZURE_ENDPOINT', None),
            api_version=os.environ.get('OPENAI_API_VERSION', None),
            azure_deployment=os.environ.get('OPENAI_AZURE_DEPLOYMENT', None)
        )

    fns_to_load = []
    for row in tqdm(data, total=len(data)):
        dataset_idx = row['dataset_idx']
        out_fn = os.path.join(out_dir, f'{dataset_idx}.json')
        fns_to_load.append(out_fn)

        if os.path.exists(out_fn) and not args.overwrite:
            print(f'{out_fn} exists. Run with -overwrite if you want to redo it...')
        else:
            paras = re.split('\n\n', row['text'])
            paras = [p.strip() for p in paras if len(p.strip()) > 0]

            # Sometimes headers are alone. If so, group them with next paragraph
            paras = group_headers(paras, min_para_toks=args.min_para_toks)

            para_idx = int(np.random.randint(len(paras)))
            para = paras[para_idx]
            original = para

            if len(para.split(' ')) > args.max_para_toks:
                para = ' '.join(para.split(' ')[:args.max_para_toks])

            source = row['source']

            # if source == 'mimic':
            #     meta = json.loads(row['meta'])
            #     assert meta['note_type'] in {'Discharge summary', 'Radiology'}
            #     source_doc_desc = meta['note_type']
            #     if meta['note_type'] == 'Radiology':
            #         source_doc_desc += ' Report'
            # else:
            #     source_doc_desc = SOURCE_DESCRIPTIONS[source]

            # print(para)
            # print('\n')

            prompt = TEMPLATES[args.dimension].format(para)  # , source_doc_desc)

            if args.model == 'mixtral':
                label = mixtral_likert(model, tokenizer, prompt=prompt)
            else:
                label = gpt_score(args.model, prompt=prompt)

            # print(prompt)
            # print(f'Label: {label}')
            # print('\n\n\n')
            # print('*' * 50)
            # print('\n\n\n')

            out_row = {
                'uuid': row['uuid'],
                'id': row['id'],
                'source': source,
                'text': para,
                'original': original, # Pre-truncation to avoid OOM errors (usaully = para)
                'label': label,
                'paragraph_idx': para_idx,
            }

            with open(out_fn, 'w') as fd:
                json.dump(out_row, fd)

    # dataset = Dataset.from_list([
    #     json.load(open(fn, 'r')) for fn in fns_to_load
    # ])

    # hf_out = os.path.join(out_dir, 'dataset_hf')
    # print(f'Saving {len(dataset)} quality-labeled paragraphs to {hf_out}')
    # dataset.save_to_disk(hf_out)
