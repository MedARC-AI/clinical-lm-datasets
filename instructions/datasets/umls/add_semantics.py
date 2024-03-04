import os
from collections import Counter, defaultdict

import argparse
import pandas as pd


BASE_DIR = '/weka/home-griffin/clinical_pile/umls'
SEM_GROUPS_FP = os.path.join(BASE_DIR, 'sem_groups.csv')
CONCEPT_FN = 'MRCONSO.RRF'
TUI_FN = 'MRSTY.RRF'


def load_concepts(rrf_dir):
    columns = [
        'cui',
        'language',
        'ts',
        'lui',
        'stt',
        'sui',
        'ispref',
        'aui',
        'saui',
        'scui',
        'sdui',
        'sab',
        'tty',
        'code',
        'str',
        'srl',
        'suppress',
        'cvf'
    ]

    # 13,609,918 - Full UMLS
    in_fn = os.path.join(rrf_dir, CONCEPT_FN)
    print(f'Loading in CUIS from {in_fn}')
    cui_df = pd.read_csv(in_fn, names=columns, delimiter='|', index_col=False)
    print(f'{len(cui_df)} total concepts')

    # 8,603,906 rows - FULL UMLS
    eng = cui_df[cui_df['language'] == 'ENG']
    print(f'{len(eng)} English concepts')

    eng.dropna(subset=['cui'], inplace=True)

    print(f'{len(eng)} English non-null concepts')
    outputs = []

    print('De-Duping CUIs and keeping track of all unique names...')
    for cui, sub_df in eng.groupby('cui'):
        name_cts = sub_df['str'].value_counts().to_dict()
        names = list(name_cts.keys())
        cts = [name_cts[n] for n in names]
        assert max(cts) == cts[0]

        outputs.append({
            'cui': cui,
            'name': names[0],
            'names': '|'.join(names),
            'name_freqs': '|'.join(map(str, cts))
        })

    outputs = pd.DataFrame(outputs)
    print(f'Returning {len(outputs)} *UNIQUE* CUIs')
    return outputs


def load_cui2tui(rrf_dir):
    columns = [
        'cui',
        'tui',
        'stn',
        'sty',
        'atui',
        'cvf'
    ]

    in_fn = os.path.join(rrf_dir, TUI_FN)
    print(f'Loading in TUI information from {in_fn}')
    tui_df = pd.read_csv(in_fn, names=columns, delimiter='|', index_col=False)

    cui2tui = defaultdict(set)
    tui2name = defaultdict(str)

    for record in tui_df.to_dict('records'):
        cui2tui[record['cui']].add(record['tui'])
        tui2name[record['tui']] = record['sty']

    return cui2tui, tui2name


def create_tui2sg_dict():
    print(f'Loading in semantic groups...')
    semgroups = pd.read_csv(SEM_GROUPS_FP, delimiter='|')
    tui2sg = dict(zip(semgroups['tui'], semgroups['sem_group_name']))
    return tui2sg


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to add metadata (TUIs, semantic groups) to raw CUIs.')

    parser.add_argument('--sources', default='all', choices=['all', 'level_0'])

    args = parser.parse_args()

    out_fn = os.path.join(BASE_DIR, args.sources, 'cui.csv')
    print(f'Will be saving output to {out_fn}')

    cui_df = load_concepts(rrf_dir=os.path.join(BASE_DIR, args.sources))
    cui2tui, tui2name = load_cui2tui(rrf_dir=os.path.join(BASE_DIR, args.sources))

    print(f'Adding TUIs for each CUI...')
    cui_df = cui_df.assign(
        tuis=cui_df['cui'].apply(lambda cui: '|'.join(list(cui2tui[cui]))),
        tui_names=cui_df['cui'].apply(lambda cui: '|'.join([tui2name[tui] for tui in list(cui2tui[cui])])),
    )

    print(f'Loading in semantic groups...')
    tui2sg = create_tui2sg_dict()

    print(f'Adding semantic groups based on TUIs...')
    cui_df = cui_df.assign(
        sem_groups=cui_df['tuis'].apply(
            lambda tuis: '|'.join(
                [x[0] for x in  Counter([tui2sg[t] for t in tuis.split('|')]).most_common()]
        ))
    )

    print(f'Selecting most common semantic group as main_sem_group...')
    cui_df = cui_df.assign(
        main_sem_group=cui_df['sem_groups'].apply(lambda x: x.split('|')[0])
    )

    print(f'Saving {len(cui_df)} CUIs to {out_fn}')
    cui_df.to_csv(out_fn, index=False)
