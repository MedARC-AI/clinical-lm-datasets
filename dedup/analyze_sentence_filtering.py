import regex as re

import gzip
import json
import os
from collections import Counter, defaultdict
from glob import glob

from datasets import load_dataset
from tqdm import tqdm


REMOVED_DIR = '/weka/home-griffin/clinical_pile/v1/dedup/sentence/removed_sentences'


REMOVED_START = "\033[91m>>>"
REMOVED_END = "<<<\u001b[0m"

S_REGEX = re.compile(re.escape(REMOVED_START))
E_REGEX = re.compile(re.escape(REMOVED_END))


if __name__ == '__main__':
    total_toks = 0
    total_removed_toks = 0

    total_toks_by_source = defaultdict(int)
    total_removed_toks_by_source = defaultdict(int)

    for fn in glob(os.path.join(REMOVED_DIR, '*.gz')):
        print(f'Analyzing removed sentences from {fn}')
        with gzip.open(fn, 'r') as fd:
            lines = fd.readlines()
            for line in tqdm(lines, total=len(lines)):
                line = line.strip()
                if len(line) == 0:
                    continue

                line = json.loads(line)
                text = line['text']
                source = line['metadata']['source']

                doc_toks = len(re.split(r'\W+', text))
                removed_toks = 0

                if REMOVED_START in text:
                    assert REMOVED_END in text
                    # start_idxs = [m.start() for m in S_REGEX.finditer(text)]
                    # end_idxs = [m.end() for m in E_REGEX.finditer(text)]
                    # for s, e in zip(start_idxs, end_idxs):
                        # print(text[s - 1:e + 1])
                        # print('\n')
                    start_idxs = [m.start() for m in S_REGEX.finditer(text)]
                    end_idxs = [m.end() for m in E_REGEX.finditer(text)]
                    for s, e in zip(start_idxs, end_idxs):
                        removed_toks += len(re.split(r'\W+', text[s:e]))

                        if source in {'medline_plus_genes'}:
                            print(source)
                            print(text[s:e])
                            print('\n\n')
                            print('*' * 50)
                            print('\n\n')

                total_toks += doc_toks
                total_toks_by_source[source] += doc_toks

                total_removed_toks += removed_toks
                total_removed_toks_by_source[source] += removed_toks

    print(f'Removed {round(total_removed_toks / total_toks, 3)} Tokens by Exact Sentence De-Dup.')
    print('\nBreaking it down by data source...')
    for source in total_toks_by_source:
        print(source)
        print(f'\t - Removed {round(total_removed_toks_by_source[source] / total_toks_by_source[source], 3)} Tokens.')

                
