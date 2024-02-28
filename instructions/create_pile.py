from datasets import DatasetDict, load_from_disk


# /weka/home-griffin/clinical_instructions/medalpaca/wikidoc_patient_hf
# /weka/home-griffin/clinical_instructions/medalpaca/wikidoc_patient_hf
# 

import os
import json
import regex as re
import argparse
from dataclasses import dataclass

from datasets import concatenate_datasets, load_from_disk
import pandas as pd


BASE_DIR = '/weka/home-griffin/clinical_instructions'
OUT_DIR = os.path.join(BASE_DIR, 'v1')
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class Source:
    name: str
    hf_path: str


SOURCES = [
    Source(name='multimedqa', hf_path='multimedqa/dataset_hf'),
    Source(name='medalpaca_flashcards', hf_path='medalpaca/flashcards_hf'),
    Source(name='medalpaca_wikidoc_patient', hf_path='medalpaca/wikidoc_patient_hf'),
    Source(name='mednli', hf_path='mednli/dataset_hf'),
    Source(name='chat_doctor', hf_path='ChatDoctor/dataset_hf'),
]


MANDATORY_COLS = [
    'id',
    'prompt',
    'completion'
]


def get_token_ct(text):
    return len(re.split(r'\W+', text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Combining individual dataset for Clinical PILE v1')

    parser.add_argument('--hub_name', default=None, type=str)

    args = parser.parse_args()

    new_splits = {'train': [], 'validation': [], 'test': []}
    stats = []
    for source in SOURCES:
        in_dir = os.path.join(BASE_DIR, source.hf_path)
        print(f'Loading {source.name} from {in_dir}')
        dataset = load_from_disk(in_dir)

        if type(dataset) != DatasetDict:
            print('\n\n')
            print('*' * 100)
            print('NO splits found. Assuming it is all for training.')
            print('*' * 100)
            print('\n\n')
            dataset = {'train': dataset}

        for split, split_data in dataset.items():
            n = len(split_data)
            split_data = split_data.filter(lambda row: len(row['id']) > 0)
            new_n = len(dataset)
            removed_n = n - new_n
            if removed_n > 0:
                print('\n\n')
                print('*' * 100)
                print(f'Removing {removed_n} examples with empty string ID. Fix this for next version.')
                print('*' * 100)
                print('\n\n')

            meta_cols = [x for x in split_data.features if x not in MANDATORY_COLS]

            assert all([type(x) == str and len(x) > 0 for x in split_data['id']])
            assert all([type(x) == str and len(x) > 0 for x in split_data['prompt']])
            assert all([type(x) == str and len(x) > 0 for x in split_data['completion']])

            assert len(split_data) == len(set(split_data['id']))

            if 'num_tokens' not in split_data.features:
                print('Adding token counts which were missing...')
                split_data = split_data.map(
                    lambda row: {'num_tokens': get_token_ct(row['prompt'])},
                    num_proc=16
                )

            if 'source' in meta_cols:
                print('Existing source will get over-written!')

            split_data = split_data.map(lambda _: {'source': source.name})

            stats.append({
                'source': source.name,
                'split': split,
                'path': in_dir,
                'tokens': sum(dataset['num_tokens']),
                'examples': len(dataset),
            })

            if len(meta_cols) > 0:
                split_data = split_data.map(
                    lambda row: {'meta': json.dumps({k: row[k] for k in meta_cols})},
                    remove_columns=meta_cols,
                    num_proc=32
                )

        new_splits[split].append(split_data)
    
    new_splits = {k: concatenate_datasets(v) for k, v in new_splits.items()}
    assert len(new_splits) == len(set(new_splits['id']))
    if args.hub_name is None:
        out_dir = os.path.join(OUT_DIR, 'dataset_hf')

        print(f'Saving {len(new_splits)} examples to {out_dir}')
        new_splits.save_to_disk(out_dir)
    else:
        print(f'Pushing PILE ({len(new_splits)}) to HuggingFace Hub --> {args.hub_name}')
        new_splits.push_to_hub(args.hub_name)
    stats = pd.DataFrame(stats)
    out_fn =  os.path.join(OUT_DIR, 'sources.csv')
    print(f'Saving information on all {len(stats)} sources in Clinical PILE to {out_fn}')
    stats.to_csv(out_fn, index=False)

    print(stats[['source', 'examples', 'tokens']].sort_values(by='tokens', ascending=False).head(n=25))