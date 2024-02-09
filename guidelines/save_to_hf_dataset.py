import json
import os
import regex as re

from datasets import Dataset, load_dataset


GUIDELINE_DIR = '/weka/home-griffin/clinical_pile/guidelines'
MIN_TOKENS = 50
OUT_DIR = '/weka/home-griffin/clinical_pile/guidelines/dataset_hf'


if __name__ == '__main__':
    existing = load_dataset('epfl-llm/guidelines')

    dataset = []

    for split in ['train', 'test', 'val']:
        fn = os.path.join(GUIDELINE_DIR, f'cleanguidelines_{split}.jsonl')
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
                        'title': obj['title'],
                        'text': obj['clean_text'].strip(),
                        'num_tokens': num_tokens,
                        'url': obj['url'],
                        'overview': obj['overview']
                    }

                    dataset.append(data_obj)
                else:
                    print(f'{num_tokens} tokens in guideline is less than minimum allowed ({MIN_TOKENS})')
    dataset = Dataset.from_list(dataset)
    print(f'Saving {len(dataset)} guidelines to {OUT_DIR}')
    dataset.save_to_disk(OUT_DIR)

