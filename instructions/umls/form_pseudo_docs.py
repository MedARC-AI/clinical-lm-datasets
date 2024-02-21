import os
import regex as re

import pandas as pd
from datasets import Dataset
from tqdm import tqdm

BASE_DIR = '/weka/home-griffin/clinical_pile/umls'
IN_FN = os.path.join(BASE_DIR, 'all', 'cui_definitions_and_relations.csv')
OUT_DIR = os.path.join(BASE_DIR, 'all', 'definitions_hf')


MIN_DEFINITION_TOK_CT = 5
MIN_DEFINITION_UNIGRAM_OVERLAP = 0.4


def filter_and_dedup(definitions):
    toks = [
        [t.strip() for t in re.split(r'\W+', d.split(':')[1].strip()) if len(t.strip()) > 0]
        for d in definitions
    ]

    tok_sets = [set(t) for t in toks]
    keep_idxs = []
    for i in range(len(definitions)):
        is_valid = len(toks[i]) >= MIN_DEFINITION_TOK_CT
        if not is_valid:
            continue

        for j in range(i):
            num = len(tok_sets[i].intersection(tok_sets[j]))
            denom = (len(tok_sets[i]) + len(tok_sets[j])) / 2.0
            overlap = num / denom

            if MIN_DEFINITION_UNIGRAM_OVERLAP > 0.4:
                print(overlap)
                print(definitions[i])
                print(definitions[j])
                print('\n\n\n\n\n')
                is_valid = False
                break
        if is_valid:
            keep_idxs.append(i)

    return [definitions[i] for i in keep_idxs]


def remove_urls(text):
    # Regular expression to match URLs
    url_pattern = re.compile(r'"?https?://\S+"?|"?www\.\S+"?')
    # Replace URLs with an empty string
    return url_pattern.sub('', text)

def remove_html(text):
    # Regular expression to match HTML tags
    url_pattern = re.compile(r'<\/?[a-z]+>')
    # Replace URLs with an empty string
    return url_pattern.sub('', text)


def form_prompt(row):
    cui = row['cui']
    tui_str = row['tui_names'].replace('|', ', ')
    name = row['name']
    definitions_str = remove_html(remove_urls(row['definitions']).strip())
    definitions_str = re.sub(r'\[ ?pmid:\d+ ?\]', '', definitions_str, flags=re.IGNORECASE)
    definitions_str = re.sub(r' {2,}', ' ', definitions_str)

    definitions = '\n'.join(filter_and_dedup(definitions_str.split('\n'))).strip()
    if len(definitions) == 0:
        return None

    text = f'# Medical Concept: {name}\n\n## Semantic Types: {tui_str}\n\n## Definitions:\n{definitions}'.strip()

    num_tokens = len(re.split(r'\W+', text))

    return {
        'id': cui,
        'text': text,
        'num_tokens': num_tokens
    }


if __name__ == '__main__':
    print(f'Reading in CUI information from {IN_FN}')
    df = pd.read_csv(IN_FN)

    records = df.dropna(subset=['definitions']).to_dict('records')

    outputs = Dataset.from_list(list(filter(None, list(tqdm(map(form_prompt, records), total=len(records))))))

    print(f'Saving {len(outputs)} CUIs to {OUT_DIR}')
    outputs.save_to_disk(OUT_DIR)
