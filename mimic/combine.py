from datasets import Dataset
import os

import pandas as pd
import regex as re

from constants import MIMIC_DIR, MIMIC_III_NOTES, MIMIC_IV_DSUMS, MIMIC_IV_REPORTS
from utils import clean_mimic


WHITELIST_CATEGORIES = [
    'Discharge summary',
    'Radiology',
]

FILTERED_NOTES_DIR = os.path.join(MIMIC_DIR, 'combined_hf')
MIN_TOKENS = 100


def compute_token_count(text):
    return len(re.split(r'\W+', text))


if __name__ == '__main__':
    print(f'Reading in notes from {MIMIC_III_NOTES}...')
    mimic_iii = pd.read_csv(MIMIC_III_NOTES).dropna(subset=['HADM_ID', 'CATEGORY', 'SUBJECT_ID', 'TEXT'])
    mimic_iii.drop_duplicates(subset='TEXT', inplace=True)
    mimic_iii = mimic_iii[mimic_iii['ISERROR'].isnull()]
    # some categories have trailing spaces
    mimic_iii['CATEGORY'] = mimic_iii['CATEGORY'].apply(lambda cat: cat.strip())

    outputs = []
    for category in WHITELIST_CATEGORIES:
        sub = mimic_iii[mimic_iii['CATEGORY'] == category].reset_index(drop=True)
        print(f'Adding {len(sub)} {category} notes from MIMIC-III')
        for row in sub.to_dict('records'):
            out_row = {
                'id': row['ROW_ID'],
                'version': 'mimic_iii',
                'note_type': row['CATEGORY'],
                'patient_id': str(round(row['SUBJECT_ID'])),
                'visit_id': str(round(row['HADM_ID'])),
                'text': row['TEXT'].strip(),
                'date': row['CHARTDATE'],
                'time': row['CHARTTIME'],
            }

            outputs.append(out_row)

    mimic_iv_dsums = pd.read_csv(MIMIC_IV_DSUMS)
    mimic_iv_dsums.drop_duplicates(subset='text', inplace=True)

    for dsum in mimic_iv_dsums.to_dict('record'):
        out_row = {
            'id': dsum['note_id'],
            'version': 'mimic_iv',
            'note_type': 'Discharge Summary',
            'patient_id': out_row['subject_id'],
            'visit_id': out_row['hadm_id'],
            'text': out_row['text'],
            'time': out_row['charttime'],
            'note_seq': out_row['note_seq']  # What is this?
        }

        outputs.append(out_row)

    mimic_iv_reports = pd.read_csv(MIMIC_IV_REPORTS)
    mimic_iv_reports.drop_duplicates(subset='text', inplace=True)

    for report in mimic_iv_reports.to_dict('record'):
        out_row = {
            'id': dsum['note_id'],
            'version': 'mimic_iv',
            'note_type': 'Radiology',
            'patient_id': out_row['subject_id'],
            'visit_id': out_row['hadm_id'],
            'text': out_row['text'],
            'time': out_row['charttime'],
            'note_seq': out_row['note_seq']  # What is this?
        }

        outputs.append(out_row)

    outputs = Dataset.from_list(outputs)

    # Including token counts
    outputs = outputs.map(
        lambda row: {
            'num_tokens': compute_token_count(row['text']),
            'text': clean_mimic(row['text'], has_identifiers=row['version'] == 'mimic_iii')
        },
        num_proc=16
    )

    prev_n = len(outputs)
    outputs = outputs.filter(lambda row: row['num_tokens'] >= MIN_TOKENS)
    n = len(outputs)
    print(f'Removed {prev_n - n} notes from Min Token filter.')

    print(f'Saving {n} notes to {FILTERED_NOTES_DIR}...')
    outputs.save_to_disk(FILTERED_NOTES_DIR)
