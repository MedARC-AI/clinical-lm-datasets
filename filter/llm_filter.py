import os
import regex as re

import argparse
import pandas as pd
import torch
from datasets import load_from_disk, load_dataset
from openai import AzureOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_TEMPLATE = ("""
<<<
{}
>>>

The previous paragraph demarcated within <<< and >>> comes from a Discharge Summary.

Your job is to assess its educational value to a medical student or resident.

A paragraph with high educational value:
1) Is clearly formatted and non-redundant.
2) Is dense with medical concepts.
3) Contains specific, non-trivial information about these concepts.
4) If about a patient, is necessary to understand the patient's condition or care plan.
5) Can be any length.

Answer with a number from 1 (no educational value) to 5 (high educational value).

DO NOT provide an explanation.
""")


ANSWER_PREFIX = 'The Answer is'
MIXTRAL_TEMPERATURE = 0.1

Y_IDX = 627
N_IDX = 418


MODELS = {
    'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'gpt-4': 'gpt-4'
}


def gpt_yes_prob(prompt, model='gpt-4', temperature=0, max_tokens=2048):
    messages = [
        {'role': 'system', 'content': 'You are a professor at a top medical school and identify highly useful reading material for your students.'},
        {'role': 'user', 'content': prompt}
    ]

    completion = client.with_options(max_retries=5).chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    raw = completion.choices[0].message.content
    assert raw.lower() in {'y', 'n'}
    return 1 if raw.lower() == 'y' else 0


def get_logits(model, tokenizer, prompt):
    messages = [
        {'role': 'user', 'content': prompt},
    ]

    chat_text = tokenizer.apply_chat_template(messages, tokenize=False) + '\n' + ANSWER_PREFIX
    inputs = tokenizer(chat_text, return_tensors='pt')['input_ids'].to('cuda')
    logits = model(inputs).logits[:, -1, :][0]
    ans_probs = torch.softmax(MIXTRAL_TEMPERATURE * logits[[Y_IDX, N_IDX]], dim=0)
    y_prob = ans_probs[0]
    return y_prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running LLM-based quality filter on paragraphs for different datasets.')
    
    parser.add_argument('--model', default='mixtral', choices=list(MODELS.keys()))

    args = parser.parse_args()

    data = load_from_disk('/weka/home-griffin/clinical_pile/v1/sample_hf')

    good_paragraph = "Allergy is a disorder of the immune system that is often called atopy. Allergic reactions occur to environmental substances known as allergens; these reactions are acquired, predictable and rapid. Strictly, allergy is one of four forms of hypersensitivity and is called type I (or immediate) hypersensitivity. It is characterized by excessive activation of certain white blood cells called mast cells and basophils by a type of antibody, known as IgE, resulting in an extreme inflammatory response. Common allergic reactions include eczema, hives, hay fever, asthma, food allergies, and reactions to the venom of stinging insects such as wasps and bees."
    hpi = "Patient is a 48 year-old well-nourished Hispanic male with a 2-month history of Rheumatoid Arthritis and strong family history of autoimmune diseases presenting after an episode of lightheadedness and muscle weakness."
    bad_paragraph = "References Kay AB (2000). Overview of 'allergy and allergic diseases: with a view to the future. Br. Med. Bull. 56 (4): 843–64. PMID 11359624. Template:WikiDoc Sources"
    other_bad = "'Open Arms' is a song by American singer-songwriter SZA (pictured) from her second studio album, SOS (2022), featuring American rapper Travis Scott. It is one of the album's guitar-backed acoustic ballads, exploring a style of music that departs from SZA's usual R&B-leaning sound."

    if args.model == 'mixtral':
        model = AutoModelForCausalLM.from_pretrained(
            MODELS[args.model],
            torch_dtype='auto',
            attn_implementation='flash_attention_2',
            device_map='auto',
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        assert args.model == 'gpt-4'
        client = AzureOpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
            azure_endpoint=os.environ.get('OPENAI_AZURE_ENDPOINT', None),
            api_version=os.environ.get('OPENAI_API_VERSION', None),
            azure_deployment=os.environ.get('OPENAI_AZURE_DEPLOYMENT', None)
        )

    for row in data:
        paras = re.split('\n\n', row['text'])
        for para in paras:
            print(para)
            print('\n')
            if args.model == 'mixtral':
                y_prob = get_logits(model, tokenizer, PROMPT_TEMPLATE.format(para))
            else:
                y_prob = gpt_yes_prob(args.model, prompt=PROMPT_TEMPLATE.format(para))
            print('\n\n\n')
            print('*' * 50)
            print('\n\n\n')
