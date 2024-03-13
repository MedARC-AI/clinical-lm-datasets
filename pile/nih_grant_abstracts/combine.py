import regex as re
import os
from glob import glob
from datasets import Dataset

import pandas as pd

NIH_DIR = '/weka/home-griffin/clinical_pile/nih_grant_abstracts'
os.makedirs(NIH_DIR, exist_ok=True)
NIH_DIR_HF = os.path.join(NIH_DIR, 'dataset_hf')
if os.path.exists(NIH_DIR_HF):
    print(f'{NIH_DIR_HF} already exists. First run \"rm -rf {NIH_DIR_HF}\"')
    exit(0)

YEARS = list(range(1985, 2023))
MIN_ABSTRACT_TOKENS = 50


if __name__ == '__main__':
    fns = list(sorted(glob(os.path.join(NIH_DIR, '*.csv'))))

    assert len(fns) == len(YEARS)
    combined_rows = []
    total_token_ct = 0
    too_short = 0
    total_ct = 0

    for year, fn in zip(YEARS, fns):
        assert f'FY{year}' in fn
        print(f'Processing {fn}...')
        try:
            df = pd.read_csv(fn)
        except:
            # Encoding errors in some of these file
            print(f'Could not read in {fn} regularly...')
            df = pd.read_csv(open(fn, encoding='utf8', errors='backslashreplace'))
        for row in df.drop_duplicates().dropna().to_dict('records'):
            abstract = row['ABSTRACT_TEXT']
            abstract = abstract.replace('[unreadable]', '')
            abstract = abstract.replace('DESCRIPTION (provided by applicant)', '')
            abstract = re.sub('\s+', ' ', abstract).strip()

            total_ct += 1
            num_tokens = len(re.split('\W+', abstract))

            if num_tokens >= MIN_ABSTRACT_TOKENS:
                combined_rows.append({
                    'id': str(year) + '_'+ str(row['APPLICATION_ID']),
                    'num_tokens': num_tokens,
                    'text': abstract
                })
                total_token_ct += num_tokens
            else:
                too_short += 1

    print(f'{too_short}/{total_ct} abstracts had fewer than {MIN_ABSTRACT_TOKENS} tokens.')

    full_df = pd.DataFrame(combined_rows)
    n = len(full_df)

    full_df = full_df.drop_duplicates(subset=['text'])
    deduped_n = len(full_df)
    print(f'{deduped_n} / {n} left after removing exact match duplicate abstracts.')

    dataset = Dataset.from_list(full_df.to_dict('records'))
    print(f'Saving {len(dataset)} rows ')
    dataset.save_to_disk(NIH_DIR_HF)
