import json
import os
import regex as re

from datasets import Dataset, load_dataset


GUIDELINE_DIR = '/weka/home-griffin/clinical_pile/guidelines'
MIN_TOKENS = 50
OUT_DIR = '/weka/home-griffin/clinical_pile/guidelines/dataset_hf'


if __name__ == '__main__':
    existing = load_dataset('epfl-llm/guidelines', split='train')

    dataset = []

    all_texts = set()

    for row in existing:
        num_tokens = len(re.split(r'\W+', row['clean_text'].strip()))

        if num_tokens >= MIN_TOKENS:
            assert row['clean_text'].strip() not in all_texts
            all_texts.add(row['clean_text'].strip())
            dataset.append({
                'id': row['id'],
                'guideline_source': row['source'],
                'accessed_from_hf': True,
                'title': row['title'],
                'text': row['clean_text'].strip(),
                'num_tokens': num_tokens,
                'url': row['url'],
                'overview': row['overview']
            })
        else:
            print(f'{num_tokens} tokens in guideline is less than minimum allowed ({MIN_TOKENS})')

    fn = os.path.join(GUIDELINE_DIR, f'cleanguidelines.jsonl')
    print(f'Reading in {fn}')
    with open(fn, 'r') as fd:
        lines = fd.readlines()
        objs = [json.loads(line.strip()) for line in lines if len(line.strip()) > 0]

        for obj in objs:
            num_tokens = len(re.split(r'\W+', obj['clean_text'].strip()))
            if num_tokens >= MIN_TOKENS:
                data_obj = {
                    'id': obj['id'],
                    'guideline_source': obj['source'], 
                    'accessed_from_hf': False,
                    'title': obj['title'],
                    'text': obj['clean_text'].strip(),
                    'num_tokens': num_tokens,
                    'url': obj['url'],
                    'overview': obj['overview']
                }
                if row['clean_text'].strip() in all_texts:
                    print('Removing duplicate from ' + obj['source'])
                else:
                    all_texts.add(row['clean_text'].strip())
                    dataset.append(data_obj)
            else:
                print(f'{num_tokens} tokens in guideline is less than minimum allowed ({MIN_TOKENS})')
    dataset = Dataset.from_list(dataset)
    print(f'Saving {len(dataset)} guidelines to {OUT_DIR}')
    assert len(dataset) == len(set(dataset['id']))
    dataset.save_to_disk(OUT_DIR)
