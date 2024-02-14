from datasets import Dataset, concatenate_datasets
import os

import numpy as np
import pandas as pd
import regex as re

from constants import MIMIC_DIR, MIMIC_III_NOTES, MIMIC_IV_DSUMS, MIMIC_IV_REPORTS
from utils import clean_mimic


WHITELIST_CATEGORIES = [
    'Discharge summary',
    # 'Radiology',
]

FILTERED_NOTES_DIR = os.path.join(MIMIC_DIR, 'combined_hf')
DEBUG_DIR = os.path.join(MIMIC_DIR, 'mini_hf')
MIN_TOKENS = 100


def compute_token_count(text):
    return len(re.split(r'\W+', text))


def sample(hf, n):
    if n >= len(hf):
        return hf
    idxs = np.arange(len(hf))
    np.random.shuffle(idxs)
    return hf.select(idxs[:n])


if __name__ == '__main__':
    print(f'Reading in MIMIC-III notes from {MIMIC_III_NOTES}...')
    mimic_iii = pd.read_csv(MIMIC_III_NOTES)
    mimic_iii = mimic_iii.dropna(subset=['CATEGORY', 'SUBJECT_ID', 'TEXT']).fillna('N/A').drop_duplicates(subset='TEXT')
    mimic_iii = mimic_iii[mimic_iii['ISERROR'] == 'N/A']
    # some categories have trailing spaces
    mimic_iii['CATEGORY'] = mimic_iii['CATEGORY'].apply(lambda cat: cat.strip())

    outputs = []
    for category in WHITELIST_CATEGORIES:
        sub = mimic_iii[mimic_iii['CATEGORY'] == category].reset_index(drop=True)
        print(f'Adding {len(sub)} {category} notes from MIMIC-III')
        for row in sub.to_dict('records'):
            out_row = {
                'id': str(round(row['ROW_ID'])),
                'version': 'mimic_iii',
                'note_type': row['CATEGORY'],
                'patient_id': str(round(row['SUBJECT_ID'])),
                'visit_id': row['HADM_ID'] if type(row['HADM_ID']) == str else str(round(row['HADM_ID'])),
                'text': row['TEXT'].strip(),
                'date': row['CHARTDATE'],
                'time': str(row['CHARTTIME']),
            }

            outputs.append(out_row)

    print(f'Reading in MIMIC-IV Discharge summaries from {MIMIC_IV_DSUMS}')
    mimic_iv_dsums = pd.read_csv(MIMIC_IV_DSUMS)
    tmp1 = len(mimic_iv_dsums)
    print(f'Loaded {tmp1} MIMIC-IV Discharge summaries from {MIMIC_IV_DSUMS}')
    mimic_iv_dsums = mimic_iv_dsums.dropna(subset=['text']).drop_duplicates(subset='text').fillna('N/A')
    tmp2 = len(mimic_iv_dsums)
    print(f'{tmp1 - tmp2} MIMIC-IV Discharge summaries either had duplicate text or no text.')

    for row in mimic_iv_dsums.to_dict('records'):
        out_row = {
            'id': row['note_id'],
            'version': 'mimic_iv',
            'note_type': 'Discharge summary',
            'patient_id': str(round(row['subject_id'])),
            'visit_id': row['hadm_id'] if type(row['hadm_id']) == str else str(round(row['hadm_id'])),
            'text': row['text'],
            'time': str(row['charttime']),
        }

        outputs.append(out_row)

    print(f'Reading in MIMIC-IV Radiology Reports from {MIMIC_IV_REPORTS}')
    mimic_iv_reports = pd.read_csv(MIMIC_IV_REPORTS)
    tmp1 = len(mimic_iv_reports)
    print(f'Loaded {tmp1} MIMIC-IV Radiology Reports from {MIMIC_IV_REPORTS}')
    mimic_iv_reports = mimic_iv_reports.dropna(subset=['text']).drop_duplicates(subset='text').fillna('N/A')
    tmp2 = len(mimic_iv_reports)
    print(f'{tmp1 - tmp2} MIMIC-IV Radiology Reports either had duplicate text or no text.')

    for row in mimic_iv_reports.to_dict('records'):
        out_row = {
            'id': row['note_id'],
            'version': 'mimic_iv',
            'note_type': 'Radiology',
            'patient_id': str(round(row['subject_id'])),
            'visit_id': row['hadm_id'] if type(row['hadm_id']) == str else str(round(row['hadm_id'])),
            'text': row['text'],
            'time': str(row['charttime']),
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

    # Save debug dataset of 10 notes of each type
    mimic_iii_dsums = outputs.filter(lambda row: row['version'] == 'mimic_iii' and row['note_type'] == 'Discharge summary')
    mimic_iv_dsums = outputs.filter(lambda row: row['version'] == 'mimic_iv' and row['note_type'] == 'Discharge summary')
    
    # mimic_iii_reports = outputs.filter(lambda row: row['version'] == 'mimic_iii' and row['note_type'] == 'Radiology')
    mimic_iv_reports = outputs.filter(lambda row: row['version'] == 'mimic_iv' and row['note_type'] == 'Radiology')

    debug = concatenate_datasets([
        sample(mimic_iii_dsums, 10),
        sample(mimic_iv_dsums, 10),
        # sample(mimic_iii_reports, 10),
        sample(mimic_iv_reports, 10),
    ])
    
    print(f'Saving {len(debug)} examples for debugging to {DEBUG_DIR}')
    debug.save_to_disk(DEBUG_DIR)
