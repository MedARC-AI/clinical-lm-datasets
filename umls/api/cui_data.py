import os
import requests
from collections import Counter

import argparse
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_uimap
from datasets import Dataset


CUI_FN = '/weka/home-griffin/clinical_pile/umls/cuis.csv'
# English sources are listed here: https://www.nlm.nih.gov/research/umls/sourcereleasedocs/index.html
ENGLISH_SOURCES_FN = '/weka/home-griffin/clinical_pile/umls/english_sources.csv'
BASE_URI = 'https://uts-ws.nlm.nih.gov/rest'
SEM_GROUPS_FN = '/weka/home-griffin/clinical_pile/umls/sem_groups.csv'

df = pd.read_csv(ENGLISH_SOURCES_FN)
ENGLISH_SOURCES = dict(zip(df['Source'], df['Name']))


def fetch_results(url):
    key = os.environ['UMLS_API_KEY']
    if '?' not in url:
        url += '?'
    else:
        url += '&'
    url += f'apiKey={key}'
    response = requests.get(url)
    return response.json()['result']


def add_data_to_cui(row, tui2sg):
    cui = row['cui']
    concept_query = BASE_URI + f'/content/current/CUI/{cui}'
    cui_obj = fetch_results(concept_query)

    if cui_obj['definitions'] == 'NONE':
        definitions_str = ''
    else:
        definition_json = fetch_results(cui_obj['definitions'])
        definitions = []
        for d in definition_json:
            if d['rootSource'] in ENGLISH_SOURCES:
                definitions.append(ENGLISH_SOURCES[d['rootSource']] + ': ' + d['value'])
    
        definitions_str = '\n'.join(definitions)

    name = cui_obj['name']

    tuis = []
    tui_names = []

    for st in cui_obj['semanticTypes']:
        tuis.append(st['uri'].split('/')[-1].strip())
        tui_names.append(st['name'])
    
    sem_groups = Counter([tui2sg[t] for t in tuis])

    most_common_sgs = sem_groups.most_common()
    all_sem_groups = '|'.join([x[0] for x in most_common_sgs])
    main_sem_group = most_common_sgs[0][0]

    if cui_obj['definitions'] == 'NONE':
        definitions = None
    else:
        definitions = fetch_results(cui_obj['definitions'])

    # What is an atom?
    # atom_resp = fetch_results(cui_obj['defaultPreferredAtom'])
    if cui_obj['relations'] == 'NONE':
        relation_str = ''
    else:
        def form_relation(relation):        
            if 'relatedIdName' in relation:
                return name + ' ' + relation['additionalRelationLabel'].replace('_', ' ').strip() + ' ' + relation['relatedIdName']
            return None

        relations = fetch_results(cui_obj['relations'])
        eng_relations = [x for x in relations if x['rootSource'] in ENGLISH_SOURCES]
        named_relations = [x for x in eng_relations if len(x['additionalRelationLabel']) > 0]
        relation_str = '\n'.join(list(filter(None, list(map(form_relation, named_relations)))))
    
    out_row = {
        'cui': cui,
        'name': name,
        'tuis': tuis,
        'tui_names': tui_names,
        'main_sem_group': main_sem_group,
        'all_sem_groups': all_sem_groups,
        'definitions': definitions_str,
        'relations': relation_str,
    }

    row.update(out_row)
    return row


def is_english(source):
    return source in ENGLISH_SOURCES


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to add definitions and relations to UMLS concepts (CUIs)')
    
    parser.add_argument('-debug', default=False, action='store_true')
    
    args = parser.parse_args()

    print(f'Loading in semantic groups...')
    semgroups = pd.read_csv(SEM_GROUPS_FN, delimiter='|')
    tui2sg = dict(zip(semgroups['tui'], semgroups['sem_group_name']))

    # /content/current/CUI/C0009044	Retrieves CUI
    print(f'Reading in CUIs from {CUI_FN}')
    cuis = pd.read_csv(CUI_FN)
    print(f'Loaded {len(cuis)} CUIs')

    records = cuis.to_dict('records')
    if args.debug:
        dataset = list(filter(None, list(tqdm(map(lambda record: add_data_to_cui(record, tui2sg), records)))))
    else:
        dataset = list(filter(None, list(p_uimap(lambda record: add_data_to_cui(record, tui2sg), records, num_cpus=32))))
    dataset = Dataset.from_list(dataset)
    
    out_dir = '/weka/home-griffin/clinical_pile/umls/cui_info_hf'
    print(f'Saving information (definitions, relations) for {len(dataset)} CUIs to {out_dir}')
    dataset.save_to_disk(out_dir)
