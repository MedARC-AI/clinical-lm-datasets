import os
import json
import regex as re
import argparse
from dataclasses import dataclass

from datasets import concatenate_datasets, load_from_disk, load_dataset
import pandas as pd


BASE_DIR = '/weka/home-griffin/clinical_pile'
OUT_DIR = os.path.join(BASE_DIR, 'v1')
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class Source:
    name: str
    hf_path: str


SOURCES = [
    Source(name='pubmed', hf_path='/weka/home-griffin/clinical_pile/pubmed/s2orc/s2orc-PubMed_processed.jsonl'),
    Source(name='pes2o', hf_path='pes2o/dataset_sample_hf'),
    Source(name='refined_web', hf_path='refined_web/dataset_hf'),
    Source(name='mimic', hf_path='mimic/dataset_hf'),
    Source(name='nih_grant_abstracts', hf_path='nih_grant_abstracts/dataset_hf'),
    Source(name='guidelines', hf_path='guidelines/dataset_hf'),
    Source(name='wikipedia', hf_path='wikipedia/dataset_hf'),
    Source(name='gutenberg_books', hf_path='/weka/home-griffin/clinical_pile/books/dataset_hf'),
    Source(name='chemsum', hf_path='chemistry/dataset_hf'),
    Source(name='medline_plus_genes', hf_path='medline/genes_hf'),
    Source(name='medline_plus_genetic_conditions', hf_path='medline/genetic_conditions_hf'),
    Source(name='medline_plus_medical_tests', hf_path='medline/medical_tests_hf'),
    Source(name='medline_plus_topic_summaries', hf_path='medline/topic_summaries_hf'),
    Source(name='wikidoc', hf_path='wikidoc/dataset_hf'),
    Source(name='ncbi_bookshelf', hf_path='ncbi_bookshelf/dataset_hf'),
]


MANDATORY_COLS = [
    'id',
    'text',
    'num_tokens'
]


def get_token_ct(text):
    return len(re.split(r'\W+', text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Combining individual dataset for Clinical PILE v1')

    parser.add_argument('--hub_name', default=None, type=str)

    args = parser.parse_args()

    all_datasets = []
    stats = []
    for source in SOURCES:
        in_dir = os.path.join(BASE_DIR, source.hf_path)
        print(f'Loading {source.name} from {in_dir}')

        if in_dir.endswith('json') or in_dir.endswith('jsonl'):
            hf_dataset = load_dataset('json', data_files=in_dir)['train']
        else:
            dataset = load_from_disk(in_dir)
        n = len(dataset)
        dataset = dataset.filter(lambda row: len(row['id']) > 0)
        new_n = len(dataset)
        removed_n = n - new_n
        if removed_n > 0:
            print('\n\n')
            print('*' * 100)
            print(f'Removing {removed_n} examples with empty string ID. Fix this for next version.')
            print('*' * 100)
            print('\n\n')

        meta_cols = [x for x in dataset.features if x not in MANDATORY_COLS]

        assert all([type(x) == str and len(x) > 0 for x in dataset['id']])
        assert all([type(x) == str and len(x) > 0 for x in dataset['text']])

        assert len(dataset) == len(set(dataset['id']))

        if 'num_tokens' not in dataset.features:
            print('Adding token counts which were missing...')
            dataset = dataset.map(
                lambda row: {
                    'num_tokens': get_token_ct(row['text']),
                    'uuid': row['source'] + '-' + row['id'],  # In case of overlapping ids across sources
                },
                num_proc=16
            )

        if 'source' in meta_cols:
            print('Existing source will get over-written!')

        dataset = dataset.map(lambda _: {'source': source.name})

        stats.append({
            'source': source.name,
            'path': in_dir,
            'tokens': sum(dataset['num_tokens']),
            'examples': len(dataset),
        })

        if len(meta_cols) > 0:
            dataset = dataset.map(
                lambda row: {'meta': json.dumps({k: row[k] for k in meta_cols})},
                remove_columns=meta_cols,
                num_proc=32
            )

        all_datasets.append(dataset)
    
    all_datasets = concatenate_datasets(all_datasets)
    print('Checking to make sure that there are no overlapping IDs across datasets.')
    assert len(all_datasets) == len(set(all_datasets['id']))

    if args.hub_name is None:
        out_dir = os.path.join(OUT_DIR, 'dataset_hf')

        print(f'Saving {len(all_datasets)} examples to {out_dir}')
        all_datasets.save_to_disk(out_dir)
    else:
        print(f'Pushing PILE ({len(all_datasets)}) to HuggingFace Hub --> {args.hub_name}')
        all_datasets.push_to_hub(args.hub_name)
    stats = pd.DataFrame(stats)
    out_fn =  os.path.join(OUT_DIR, 'sources.csv')
    print(f'Saving information on all {len(stats)} sources in Clinical PILE to {out_fn}')
    stats.to_csv(out_fn, index=False)

    print(stats[['source', 'examples', 'tokens']].sort_values(by='tokens', ascending=False).head(n=25))
