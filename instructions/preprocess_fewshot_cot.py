from dataclasses import dataclass
import json
import os

import argparse
from datasets import load_dataset
from openai import AzureOpenAI
import numpy as np
np.random.seed(1992)
from textwrap import dedent
import regex as re
from tqdm import tqdm


def chatgpt(client, messages, model='gpt-4', temperature=0.1, max_tokens=2048):
    completion = client.with_options(max_retries=5).chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


def build_prompt(doc, dataset):
    """
    Prompt is taken from MedPrompt
    https://github.com/microsoft/promptbase/blob/90fe3f1e2476638ae7e623687bfe9b8b2077b2bb/src/promptbase/drop/drop.py#L98
    """

    if dataset == 'pubmedqa':
        input_type = 'PubMed Abstract'
        question = doc['QUESTION']
        ctx_lines = []
        assert len(doc['LABELS']) == len(doc['CONTEXTS']) 
        for header, ctx in zip(doc['LABELS'], doc['CONTEXTS']):
            ctx_lines.append(f'# {header}: {ctx}')
        input = '\n'.join(ctx_lines)

        choices = ['yes', 'no', 'maybe']
        letters = ['A', 'B', 'C']
        choice_str = '\n'.join([f'{l}) {c}' for l, c in zip(letters, choices)])
        choice_letter_str = 'A, B, or C'

        prompt = dedent(f"""
        Answer the following medical reading comprehension **Question** based on the **{input_type}** below.
        First, think step by step and write an **Explanation** for reasoning through the question.
        Then, analyze your explanation and write just the Letter ({choice_letter_str}) corresponding to your **Final Answer**.
        ----
        **{input_type}:**\n{input}
        ----
        **Question:** {question}
        ----
        **Choices:**\n{choice_str}\n
        ----
        **Explanation**: """
        )
    elif dataset == 'medmcqa':
        choice_options = [
            doc['opa'],
            doc['opb'],
            doc['opc'],
            doc['opd'],
        ]

        question = doc['question']

        choice_str = []
        choice_letters = ['A', 'B', 'C', 'D']
        for l, o in zip(choice_letters, choice_options):
            choice_str.append(f'{l}) {o}')
        choice_str = '\n'.join(choice_str)
        choice_letter_str = ', '.join(choice_letters)

        prompt = dedent(f"""
        Answer the following medical reading comprehension **Question**.
        First, think step by step and write an **Explanation** for reasoning through the question.
        Then, analyze your explanation and write just the Letter ({choice_letter_str}) corresponding to your **Final Answer**.
        ----
        **Question:** {question}
        ----
        **Choices:**\n{choice_str}
        ----
        **Explanation**: """
        )
    elif dataset == 'medqa':
        question = doc['sent1']
        choice_letters = ['A', 'B', 'C', 'D']

        choice_options = [
            doc['ending0'],
            doc['ending1'],
            doc['ending2'],
            doc['ending3'],
        ]

        choice_str = []
        for l, o in zip(choice_letters, choice_options):
            choice_str.append(f'{l}) {o}')
        choice_str = '\n'.join(choice_str)
        choice_letter_str = ', '.join(choice_letters)

        prompt = dedent(f"""
        Answer the following reading comprehension **Question**.
        First, think step by step and write an **Explanation** for reasoning through the question.
        Then, analyze your explanation and write just the Letter ({choice_letter_str}) corresponding to your **Final Answer**.
        ----s
        **Question:** {question}
        ----
        **Choices:**\n{choice_str}
        ----
        **Explanation**: """
        )
    else:
        choice_options = doc['choices']
        choice_letters = ['A', 'B', 'C', 'D']
        question = doc['question']
        choice_str = []
        for l, o in zip(choice_letters, choice_options):
            choice_str.append(f'{l}) {o}')
        choice_str = '\n'.join(choice_str)
        choice_letter_str = ', '.join(choice_letters)

        prompt = dedent(f"""
        Answer the following reading comprehension **Question**.
        First, think step by step and write an **Explanation** for reasoning through the question.
        Then, analyze your explanation and write just the Letter ({choice_letter_str}) corresponding to your **Final Answer**.
        ----
        **Question:** {question}
        ----
        **Choices:**\n{choice_str}
        ----
        **Explanation**: """
        )

    prompt = '\n'.join([x.lstrip() for x in prompt.split('\n')])

    return prompt


def generate_self_cot(args, doc, client):
    idx = doc['idx']
    cache_fn = os.path.join(args.cache_dir, f'{idx}.json')

    if os.path.exists(cache_fn) and not args.overwrite:
        print(f'Found existing file {cache_fn} and not over-writing.')
        return None

    prompt = build_prompt(doc, args.dataset)

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant for medical question answering.'},
        {'role': 'user', 'content': prompt}
    ]
    try:
        prediction = chatgpt(client=client, messages=messages)
    except Exception as e:
        print(e)
        print('Skipping this for CoT.')
        return None

    if type(prediction) != str:
        print(f'Model returned prediction of type {type(prediction)} -> {prediction}. Returning with no CoT')
        return None

    split = re.split(re.escape('**Final Answer'), prediction)
    if len(split) != 2:
        print('Invalid output format. Check your prompt.')
        return None

    rationale, answer_raw = split
    rationale = rationale.strip()

    if args.dataset == 'pubmedqa':
        letters = ['A', 'B', 'C']
    else:
        letters = ['A', 'B', 'C', 'D']

    choice_regex = r'|'.join(letters)
    answer_match = re.search(rf'({choice_regex})', answer_raw, flags=re.IGNORECASE)
    if answer_match is None:
        print(f'Expected one of {choice_regex}. Got {answer_raw}. Check your prompt.')
        return None

    answer_lower = answer_match.group().lower()
    
    if args.dataset == 'pubmedqa':
        choices = ['yes', 'no', 'maybe']
        target = letters[choices.index(doc['final_decision'])]
    elif args.dataset == 'medqa':
        target = letters[doc['label']]
    elif args.dataset == 'medmcqa':
        target = letters[doc['cop']]
    elif args.dataset == 'mmlu':
        target = letters[doc['answer']]
    else:
        raise Exception(f'Unrecognized dataset --> {args.dataset}')

    assert target in letters
    target_lower = target.lower()

    if answer_lower != target_lower:
        print(f'Answer ({answer_lower}) didn\'t match ground truth target ({target_lower}). Not adding to CoT dataset.')
        print('Saving empty rationale to cache so we dont try this one again.')
        with open(cache_fn, 'w') as fd:
            json.dump({'rationale': ''}, fd)
    else:
        print(f'Saving rationale --> {rationale}')
        with open(cache_fn, 'w') as fd:
            json.dump({'rationale': rationale}, fd)


if __name__ == '__main__':
    # generate and cache self-COT examplars for few-shot https://github.com/microsoft/promptbase
    parser = argparse.ArgumentParser('Pre-Compute CoT rationales for MultiMedQA')

    parser.add_argument('--dataset', default='medqa')
    parser.add_argument('--cot_split', default='train')
    parser.add_argument('--max_cot_examples', default=1000, type=int)

    parser.add_argument('--cot_model', default='gpt-4')
    parser.add_argument('--cot_model_type', default='openai', choices=['openai', 'huggingface'])

    parser.add_argument('-overwrite', action='store_true', default=False)

    args = parser.parse_args()

    if args.dataset == 'medqa':
        cot_data = load_dataset('GBaker/MedQA-USMLE-4-options-hf')[args.cot_split]
    else:
        raise Exception(f'Unrecognized dataset --> {args.dataset}')

    n = len(cot_data)
    cot_data = cot_data.add_column('idx', list(range(n)))
    
    args.cache_dir = os.path.join(f'/weka/home-griffin/instructions/datasets/{args.dataset}/cot_cache')
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.max_cot_examples < n:
        data_idxs = np.arange(n)
        np.random.shuffle(data_idxs)
        sample_idxs = data_idxs[:args.max_cot_examples]
        print(f'Randomly sampling {args.max_cot_examples} examples from {n} total...')
        cot_data = cot_data.select(sample_idxs)

    assert 'OPENAI_API_KEY' in os.environ
    client = AzureOpenAI(
        api_key=os.environ.get('OPENAI_API_KEY'),
        azure_endpoint='https://east-us-2-llm.openai.azure.com/',
        api_version='2023-05-15',
        azure_deployment='misc-gpt4-turbo'
    )

    for doc in tqdm(cot_data):
        generate_self_cot(args, doc, client)
