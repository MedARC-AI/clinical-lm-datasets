import pandas as pd
import regex as re

from constants import MIMIC_III_NOTES


BLACKLIST_CATEGORIES = [
    'Nursing/other',
    'Nursing',
    'Case Management'
]

FILTERED_NOTES_FN = MIMIC_III_NOTES.replace('.csv', '') + '_note_filtered.csv'
MIN_TOKENS = 50


def compute_token_count(text):
    return len(re.split(r'\W+', text))


if __name__ == '__main__':
    print(f'Reading in notes from {MIMIC_III_NOTES}...')
    notes = pd.read_csv(MIMIC_III_NOTES)
    notes = notes[notes['ISERROR'].isnull()]
    # some categories have trailing spaces
    notes['CATEGORY'] = notes['CATEGORY'].apply(lambda cat: cat.strip()) 
    orig = len(notes)

    # Including token counts
    notes = notes.assign(token_count=notes['TEXT'].apply(compute_token_count))

    for category in BLACKLIST_CATEGORIES:
        prev_n = len(notes)
        notes = notes[notes['CATEGORY'] != category]
        n = len(notes)
        print(f'Removed {prev_n - n} notes from {category} filter.')

    # Minimum Length 50 tokens
    prev_n = len(notes)
    notes = notes[notes['token_count'] >= MIN_TOKENS]
    n = len(notes)
    # Removes 171,382 notes
    print(f'Removed {prev_n - n} notes from min token filter.')

    filt_n = len(notes)
    print(f'{orig} --> {n} notes. Saving to {FILTERED_NOTES_FN}...')
    notes.reset_index(drop=True).to_csv(FILTERED_NOTES_FN, index=False)
