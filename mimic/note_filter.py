import os

import pandas as pd
import regex as re

from constants import MIMIC_DIR, MIMIC_III_NOTES, MIMIC_IV_DSUMS, MIMIC_IV_REPORTS


WHITELIST_CATEGORIES = [
    'Discharge Summary',
    'Radiology',
]

FILTERED_NOTES_FN = os.path.join(MIMIC_DIR, 'combined.csv')
MIN_TOKENS = 100


def compute_token_count(text):
    return len(re.split(r'\W+', text))


if __name__ == '__main__':
    print(f'Reading in notes from {MIMIC_III_NOTES}...')
    mimic_iii = pd.read_csv(MIMIC_III_NOTES)
    mimic_iii = mimic_iii[mimic_iii['ISERROR'].isnull()]
    # some categories have trailing spaces
    mimic_iii['CATEGORY'] = mimic_iii['CATEGORY'].apply(lambda cat: cat.strip()) 
    orig = len(mimic_iii)

    filtered_mimic_iii = []
    for category in WHITELIST_CATEGORIES:
        sub = mimic_iii[mimic_iii['CATEGORY'] == category].reset_index(drop=True)
        print(f'Adding {len(sub)} {category} notes from MIMIC-III')
        filtered_mimic_iii.append(sub)
    
    filtered_mimic_iii = pd.concat(filtered_mimic_iii)

    mimic_iv_dsums = pd.read_csv(MIMIC_IV_DSUMS)
    mimic_iv_reports = pd.read_csv(MIMIC_IV_REPORTS)

    # Including token counts
    notes = notes.assign(token_count=notes['TEXT'].apply(compute_token_count))


    # Minimum Length 50 tokens
    prev_n = len(notes)
    notes = notes[notes['token_count'] >= MIN_TOKENS]
    n = len(notes)
    # Removes 171,382 notes
    print(f'Removed {prev_n - n} notes from min token filter.')

    filt_n = len(notes)
    print(f'{orig} --> {n} notes. Saving to {FILTERED_NOTES_FN}...')
    notes.reset_index(drop=True).to_csv(FILTERED_NOTES_FN, index=False)
