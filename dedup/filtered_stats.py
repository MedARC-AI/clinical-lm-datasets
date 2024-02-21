import gzip
import json
import os
from collections import Counter, defaultdict
from glob import glob

from datasets import load_dataset
from tqdm import tqdm


REMOVED_DIR = '/weka/home-griffin/clinical_pile/v1/dedup/minhash/removed'


if __name__ == '__main__':
    removed_ids = set()

    filtered_ids = set()
    num_removed_guidelines = 0
    for fn in glob(os.path.join(REMOVED_DIR, '*.gz')):
        print(f'Extracting removed document IDs from {fn}')
        with gzip.open(fn, 'r') as fd:
            lines = fd.readlines()
            for line in tqdm(lines, total=len(lines)):
                line = line.strip()
                if len(line) == 0:
                    continue

                line = json.loads(line)
                assert type(line) == dict
                if 'id' not in line:
                    assert line['metadata']['source'] == 'guidelines'
                    num_removed_guidelines += 1
                    # TODO This is fixed if we re-run pretrain#combine
                else:
                    if line['metadata']['source'] == 'wikidoc':
                        print(line['text'])
                        print(line['id'])
                        print('\n\n\n')
                    filtered_ids.add(line['id'])

    print(f'Num removed guidelines: {num_removed_guidelines}')
    dataset = load_dataset('medarc/clinical_pile_v1', split='train')

    all_cts = Counter(dataset['source'])

    print('Pre Filtering...')
    total = len(dataset)
    for k, v in all_cts.items():
        print(k + ' -> ' + str(round(v / total, 5)))
    print('\n\n\n')
    filtered = dataset.filter(lambda row: row['id'] in filtered_ids)

    filtered_cts = Counter(filtered['source'])

    print('Post Filtering...')
    filt_n = len(filtered)
    for k, v in filtered_cts.items():
        print(k + ' -> ' + str(round(v / filt_n, 5)))
    print('\n\n\n')

    print('Fraction Removed...')
    for k, v in all_cts.items():
        print(k + ' -> ' + str(round(filtered_cts[k] / v, 5)))
