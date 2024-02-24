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
                filtered_ids.add(line['id'])

    dataset = load_dataset('medarc/clinical_pile_v1', split='train')

    all_cts = Counter(dataset['source'])

    print('Pre Filtering...')
    total = len(dataset)
    for x in sorted(all_cts.items(), key=lambda x: (-x[1], x[0])):
        k, v = x
        print(k + ' -> ' + str(v) + ' (' + str(round(v / total, 5)) + ')')
    print('\n\n\n')

    remaining = dataset.filter(lambda row: row['id'] not in filtered_ids)
    remaining_cts = Counter(remaining['source'])

    print('Post Filtering...')
    remaining_n = len(remaining)
    for x in sorted(remaining_cts.items(), key=lambda x: (-x[1], x[0])):
        k, v = x
        print(k + ' -> ' + str(v) + ' (' + str(round(v / remaining_n, 5)) + ')')
    print('\n\n\n')

    print('Fraction Removed...')

    filtered = dataset.filter(lambda row: row['id'] in filtered_ids)
    filtered_cts = Counter(filtered['source'])

    for x in sorted(all_cts.items(), key=lambda x: (-x[1], x[0])):
        k, v = x
        print(k + ' -> ' + str(round(filtered_cts[k] / v, 5)))
