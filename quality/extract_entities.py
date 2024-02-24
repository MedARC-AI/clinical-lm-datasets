from collections import defaultdict

import argparse
import stanza
from datasets import load_dataset
from tqdm import tqdm


EXTRACTORS = {
    'chemical': 'bc4chemd', # chemicals
    'clinical': 'i2b2', # diseases, treatments, tests
    'molecular_biology': 'jnlpba',
    'diseases': 'ncbi_disease',
}


def extract_ents(text, extractor):
    ents = extractor(text).entities
    ents = [{'text': ent.text, 'type': ent.type} for ent in ents]
    return ents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and push dataset to Hugging Face Hub")
    parser.add_argument('--dataset', default='medarc/clinical_pile_v1_minhash_deduped')
    parser.add_argument('--id_col', default='id')
    parser.add_argument('--text_col', default='text')
    parser.add_argument('--extractor', default='clinical', choices=['clinical', 'biomedical'])

    args = parser.parse_args()

    extractors = {
        k: stanza.Pipeline('en', package='mimic', processors={'ner': v}, use_gpu=True) for k, v in EXTRACTORS.items()
    }

    dataset = load_dataset(args.dataset, split='train')

    for row in tqdm(dataset):
        text = row[args.text_col]
        extracted = defaultdict(list)
        for name, extractor in extractors.items():
            ents = extract_ents(text, extractor)
            for ent in ents:
                extracted[ent['text']].append(name)

        for k, v in extracted.items():
            if len(set(v)) == 1:
                print(k, v[0])
