import os

import argparse
import pandas as pd
import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_TEMPLATE = ("""
###
{}
####

Does the previous paragraph demarcated within ### and ### provide educational value for a medical student?

A datapoint with high educational value should:

1) Be clearly formatted and syntactically correct.
2) Be dense with medical concepts.
3) Contain specific, non-trivial information about these concepts.

Respond only with "Y" or "N". DO NOT provide an explanation.
""")

ANSWER_PREFIX = 'The Answer is'
TEMPERATURE = 0.1

Y_IDX = 627
N_IDX = 418


def get_logits(model, tokenizer, prompt):
    messages = [
        {'role': 'user', 'content': prompt},
    ]

    chat_text = tokenizer.apply_chat_template(messages, tokenize=False) + '\n' + ANSWER_PREFIX
    inputs = tokenizer(chat_text, return_tensors='pt')['input_ids'].to('cuda')
    logits = model(inputs).logits[:, -1, :][0]
    # mask = torch.zeros_like(logits, dtype=torch.bool)
    # mask.fill_(True)
    # mask[Y_IDX] = False
    # mask[N_IDX] = False
    # logits.masked_fill_(mask, float('-inf'))
    ans_probs = torch.softmax(TEMPERATURE * logits[[Y_IDX, N_IDX]], dim=0)
    y_prob = ans_probs[0]
    return y_prob


if __name__ == '__main__':
    data = load_from_disk('/weka/home-griffin/clinical_pile/guidelines/dataset_hf')

    good_paragraph = "Allergy is a disorder of the immune system that is often called atopy. Allergic reactions occur to environmental substances known as allergens; these reactions are acquired, predictable and rapid. Strictly, allergy is one of four forms of hypersensitivity and is called type I (or immediate) hypersensitivity. It is characterized by excessive activation of certain white blood cells called mast cells and basophils by a type of antibody, known as IgE, resulting in an extreme inflammatory response. Common allergic reactions include eczema, hives, hay fever, asthma, food allergies, and reactions to the venom of stinging insects such as wasps and bees."
    hpi = "Patient is a 48 year-old well-nourished Hispanic male with a 2-month history of Rheumatoid Arthritis and strong family history of autoimmune diseases presenting after an episode of lightheadedness and muscle weakness."
    bad_paragraph = "References Kay AB (2000). Overview of 'allergy and allergic diseases: with a view to the future. Br. Med. Bull. 56 (4): 843–64. PMID 11359624. Template:WikiDoc Sources"
    other_bad = "'Open Arms' is a song by American singer-songwriter SZA (pictured) from her second studio album, SOS (2022), featuring American rapper Travis Scott. It is one of the album's guitar-backed acoustic ballads, exploring a style of music that departs from SZA's usual R&B-leaning sound."

    # HF_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
    HF_NAME = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

    # device_map='auto'
    model = AutoModelForCausalLM.from_pretrained(
        HF_NAME,
        torch_dtype='auto',
        attn_implementation='flash_attention_2',
        device_map='auto',
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(HF_NAME)

    # print(get_logits(model, tokenizer, PROMPT_TEMPLATE.format(good_paragraph)))
    # print(get_logits(model, tokenizer, PROMPT_TEMPLATE.format(hpi)))
    # print(get_logits(model, tokenizer, PROMPT_TEMPLATE.format(bad_paragraph)))
    # print(get_logits(model, tokenizer, PROMPT_TEMPLATE.format(other_bad)))

    for row in data:
        import regex as re
        paras = re.split('\n\n', row['text'])
        for para in paras:
            print(para)
            print('\n')
            print(get_logits(model, tokenizer, PROMPT_TEMPLATE.format(para)))
            print('\n\n\n')
            print('*' * 50)
            print('\n\n\n')
