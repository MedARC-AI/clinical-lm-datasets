import os
from collections import defaultdict

import argparse
import pandas as pd


BASE_DIR = '/weka/home-griffin/clinical_pile/umls'

ENGLISH_SOURCES_FN = os.path.join(BASE_DIR, 'english_sources.csv')
_source_df = pd.read_csv(ENGLISH_SOURCES_FN)
ENGLISH_SOURCES = dict(zip(_source_df['Source'], _source_df['Name']))

REL_DESCRIPTORS_FN = os.path.join(BASE_DIR, 'relation_descriptors.csv')
_rel_descriptors_df = pd.read_csv(REL_DESCRIPTORS_FN)
REL2DESCRIPTION = dict(zip(_rel_descriptors_df['relationship_attribute'], _rel_descriptors_df['relationship_description']))


def cui2definitions(rrf_dir, cuis_to_filter_for):
    names = [
        'cui',
        'aui',
        'atui',
        'satui',
        'sab',
        'def',
        'suppress',
        'cvf'
    ]
    in_fn = os.path.join(rrf_dir, 'MRDEF.RRF')
    print(f'Loading in Definitions from {in_fn}...')
    def_df = pd.read_csv(in_fn, delimiter='|', names=names, index_col=False)
    def_df.dropna(subset=['cui', 'def', 'sab'], inplace=True)
    def_df = def_df[def_df['cui'].isin(cuis_to_filter_for)]

    eng_df = def_df[def_df['sab'].isin(ENGLISH_SOURCES)]

    cui2def = defaultdict(list)
    for row in eng_df.to_dict('records'):
        def_str = ENGLISH_SOURCES[row['sab']] + ': ' + row['def']
        cui2def[row['cui']].append(def_str)

    return def_df


def extract_relations(rrf_dir, cuis_to_filter_for, cui2name):
    names = [
        'cui1',
        'aui1',
        'stype1',
        'rel',  # Relationship of second concept or atom to first concept or atom
        'cui2',
        'aui2',
        'stype2',
        'rela',
        'rui',
        'srui',
        'sab',
        'sl',
        'rg',
        'dir',
        'suppress',
        'cvf',
    ]

    in_fn = os.path.join(rrf_dir, 'MRREL.RRF')
    print(f'Loading in Relations from {in_fn}...')
    rel_df = pd.read_csv(in_fn, delimiter='|', names=names, index_col=False)
    rel_df = rel_df[rel_df['cui1'].isin(cuis_to_filter_for)]

    rel_df = rel_df.dropna(subset=['cui1', 'cui2', 'rela'])

    rel_df = rel_df[rel_df['sab'].isin(ENGLISH_SOURCES)]
    rel_df = rel_df[rel_df['rela'].apply(lambda label: len(label) > 0)]
    rel_df = rel_df[rel_df['cui1'] != rel_df['cui2']]

    def create_prompt(row):
        a = cui2name[row.cui1]
        b = cui2name[row.cui2]
        rel_descriptor = REL2DESCRIPTION[row.rela]
        rel_str = f'{a}|{rel_descriptor}|{b}'
        return rel_str

    print('Combining relation tuples into single string...')
    rel_df['rel_prompt_args'] = rel_df.apply(create_prompt, axis=1)
    print('Combining relations for same CUI...')
    grouped = rel_df.groupby('cui1')['name'].apply(lambda x: ', '.join(x)).reset_index()
    grouped.rename(columns={'cui1': 'cui'}, inplace=True)
    return grouped[['cui', 'rel_prompt_args']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to add metadata (TUIs, semantic groups) to raw CUIs.')

    parser.add_argument('--sources', default='all', choices=['all', 'level_0'])

    args = parser.parse_args()

    rrf_dir = os.path.join(BASE_DIR, args.sources)
    in_fn = os.path.join(rrf_dir, 'cui.csv')
    out_fn = os.path.join(rrf_dir, 'cui_augmented.csv')
    print(f'Will be saving output to {out_fn}')

    print(f'Reading in CUIs from {in_fn}')
    cui_df = pd.read_csv(in_fn)
    print(f'Loaded {len(cui_df)} CUIs from {in_fn}')

    cui_set = set(cui_df['cui'])
    cui2name = dict(zip(cui_df['cui'], cui_df['name']))

    cui2defs = cui2definitions(rrf_dir, cui_set)
    relations_df = extract_relations(rrf_dir, cui_set, cui2name)

    print('Assigning definitions and relations to CUIs...')
    cui_df = cui_df.assign(
        definitions=cui_df['cui'].apply(lambda cui: '\n'.join(cui2defs.get('cui', [])).strip()),
    )

    cui_df = cui_df.join(relations_df, on='cui', how='left')

    print(f'Saving {len(cui_df)} to {out_fn}')

    def_ct = sum([1 if len(x) > 0 else 0 for x in cui_df['definitions'].str.len().tolist()])
    rel_ct = sum([1 if len(x) > 0 else 0 for x in cui_df['relations'].str.len().tolist()])

    print(f'{def_ct} CUIs have >= 1 definitions')
    print(f'{rel_ct} CUIs have >= 1 relationships')

    print(f'Saving {len(cui_df)} to {out_fn}')
    cui_df.to_csv(out_fn, index=False)
