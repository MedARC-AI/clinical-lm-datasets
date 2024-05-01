from datasets import DatasetDict, load_from_disk


import os
import json
import regex as re
from collections import Counter
import argparse
from dataclasses import dataclass

from datasets import concatenate_datasets, load_from_disk
import pandas as pd


BASE_DIR = '/weka/home-griffin/clinical_instructions'
PILE_DIR = os.path.join(BASE_DIR, 'v1')
os.makedirs(PILE_DIR, exist_ok=True)


@dataclass
class Source:
    name: str
    hf_path: str


CONFIGS = [
    Source(name='multimedqa', hf_path='multimedqa/dataset_cot_hf_artificial'),
    Source(name='medalpaca_flashcards', hf_path='medalpaca/flashcards_hf'),
    Source(name='medalpaca_wikidoc_patient', hf_path='medalpaca/wikidoc_patient_hf'),
    Source(name='mednli', hf_path='mednli/dataset_hf'),
    Source(name='chat_doctor', hf_path='ChatDoctor/dataset_hf'),
    Source(name='radqa', hf_path='radqa/dataset_hf')
]


MANDATORY_COLS = [
    'id',
    'prompt',
    'completion',
    'num_options',
    'source'
]


def get_token_ct(text):
    return len(re.split(r'\W+', text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Combining individual dataset for Clinical PILE v1')

    parser.add_argument('--output_format', default='hf', choices=['hf', 'jsonl'])
    parser.add_argument('--out_dir', default=None)

    parser.add_argument('--max_train_size', default=int(1e10), type=int)

    args = parser.parse_args()

    if args.out_dir is None:
        if args.output_format == 'jsonl':
            args.out_dir = os.path.join(PILE_DIR, 'jsonl')
        else:
            args.out_dir = os.path.join(PILE_DIR, 'dataset_hf')

    if os.path.exists(args.out_dir):
        print(f'{args.out_dir} exists already. Before re-running, run "rm -rf {args.out_dir}"')
        exit(0)

    new_splits = {'train': [], 'validation': [], 'test': []}
    stats = []
    for config in CONFIGS:
        in_dir = os.path.join(BASE_DIR, config.hf_path)
        print(f'Loading {config.name} from {in_dir}')
        dataset = load_from_disk(in_dir)

        if type(dataset) != DatasetDict:
            print('\n\n')
            print('*' * 100)
            print(f'NO splits found. Assuming {config.name} is all for training.')
            print('*' * 100)
            print('\n\n')
            dataset = {'train': dataset}

        for split, split_data in dataset.items():
            n = len(split_data)
            split_data = split_data.filter(lambda row: len(row['id']) > 0)
            new_n = len(split_data)
            removed_n = n - new_n
            if removed_n > 0:
                print('\n\n')
                print('*' * 100)
                print(f'Removing {removed_n} examples with empty string ID. Fix this for next version.')
                print('*' * 100)
                print('\n\n')

            try:
                assert all([type(x) == str and len(x) > 0 for x in split_data['id']])
                assert all([type(x) == str and len(x) > 0 for x in split_data['prompt']])
                assert all([type(x) == str and len(x) > 0 for x in split_data['completion']])
            except Exception as e:
                print(e)
                print(config.name + ' ' + split + ' failed a formatting test.')
                raise

            if len(split_data) != len(set(split_data['id'])):
                dup_ids = set([k for k, v in Counter(split_data['id']).items() if v > 1])
                print(f'{len(dup_ids)} duplicated IDs...')
                duped = split_data.filter(lambda row: row['id'] in dup_ids)
                print(dup_ids)
                print(Counter(duped['source']))
                raise Exception('Duplicated IDs. Fix first.')

            if 'num_tokens' not in split_data.features:
                print('Adding token counts which were missing...')
                split_data = split_data.map(
                    lambda row: {'num_tokens': get_token_ct(row['prompt'])},
                    num_proc=64
                )

            if 'source' in split_data.features:
                print('Existing column named source. Will not be re-writing it!')

            split_data = split_data.map(lambda row: {'source': row['source'] if 'source' in row else config.name})

            split_data_sampled = []
            for source in list(set(split_data['source'])):
                split_data_source = split_data.filter(lambda row: row['source'] == source)
                if split == 'train' and len(split_data_source) > args.max_train_size:
                    print(f'We have more training examples than allowed: {len(split_data_source)} / {args.max_train_size}')
                    print(f'Taking a random sample of {args.max_train_size}')
                    split_data_source = split_data_source.shuffle(seed=1992).select(range(args.max_train_size))
                
                stats.append({
                    'source': source,
                    'split': split,
                    'path': in_dir,
                    'tokens': sum(split_data_source['num_tokens']),
                    'examples': len(split_data_source),
                })

                split_data_sampled.append(split_data_source)

            split_data_sampled = concatenate_datasets(split_data_sampled)

            meta_cols = [x for x in split_data_sampled.features if x not in MANDATORY_COLS]

            if len(meta_cols) > 0:
                split_data_sampled = split_data_sampled.map(
                    lambda row: {'meta': json.dumps({k: row[k] for k in meta_cols})},
                    remove_columns=meta_cols,
                    num_proc=32
                )

            new_splits[split].append(split_data_sampled)

    new_splits = {k: concatenate_datasets(v) for k, v in new_splits.items()}
    
    if args.output_format == 'hf':
        new_splits = DatasetDict(new_splits)
        new_splits.save_to_disk(args.out_dir)
    elif args.output_format == 'jsonl':
        # prompt and completion
        import json
        for split, data in new_splits.items():
            out_fn = os.path.join(args.out_dir, f'{split}.jsonl')
            with open(out_fn, 'w') as fd:
                for row in data:
                    fd.write(json.dumps({
                        'prompt': row['prompt'], 'completion': 'completion',
                    }) + '\n')

    stats = pd.DataFrame(stats)
    out_fn =  os.path.join(args.out_dir, 'sources.csv')
    print(f'Saving information on all {len(stats)} sources in Clinical instruction PILE to {out_fn}')
    stats.to_csv(out_fn, index=False)

    print(stats[['source', 'split', 'examples', 'tokens']].sort_values(by='tokens', ascending=False).head(n=25))

    print('\n' + '*' * 50 + '\n')
    train_stats = stats[stats['split'] == 'train']
    print(train_stats[['source', 'split', 'examples', 'tokens']].sort_values(by='tokens', ascending=False).head(n=25))
