import orjson

from datasets import load_from_disk
from collections import defaultdict, Counter
from tqdm import tqdm

ids = defaultdict(int)

dataset = load_from_disk('/weka/home-griffin/clinical_pile/pubmed/s2orc/s2orc-PubMed_processed_hf')

id_cts = Counter(dataset['id'])
print(id_cts.most_common(n=100))

dup_ids = set()
for k, v in id_cts.items():
    if v > 1:
        dup_ids.add(k)

print(f'Duplicated Internal IDs ({len(dup_ids)})...')
print(dup_ids)

print('\n\n\n\n\n')

corpus_id_cts = Counter(dataset['meditron_corpus_id'])
print(corpus_id_cts.most_common(n=100))
dup_corpus_ids = set()
for k, v in corpus_id_cts.items():
    if v > 1:
        dup_corpus_ids.add(k)

print(f'Duplicated S2 Corpus IDs ({len(dup_corpus_ids)})...')
print(len(dup_corpus_ids))
print(dup_corpus_ids)
