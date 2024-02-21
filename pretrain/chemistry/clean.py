import os
import regex as re

from datasets import concatenate_datasets, load_dataset, Dataset
import ftfy
from tqdm import tqdm


OUT_DIR = '/weka/home-griffin/clinical_pile/chemistry'
os.makedirs(OUT_DIR, exist_ok=True)


def get_token_ct(text):
    return len(re.split(r'\W+', text))


def clean(text):
    encoded = ftfy.fix_text(text.encode().decode('unicode_escape', 'ignore'))
    encoded = encoded.replace('Å', '').replace('À', '').replace('Â', '').replace('\ue103', 'f').replace('\ue104', 'fl').replace('\ue09d', 'ft')
    encoded = re.sub(f'\[[\d,\s]+\]', ' ', encoded)
    encoded = re.sub('\s+', ' ', encoded)
    return encoded


def create_input(example):
    DELIM = '<!>'
    headers = example['headers'].split(DELIM)
    sections = example['sections'].split(DELIM)

    out_str = f'# ' + example['title'] + '\n\n'
    out_str = '## Abstract:\n\n' + example['abstract'] + '\n\n'
    for header, body in zip(headers, sections):
        if header is not None and len(header.strip()) > 0:
            out_str += '## ' + header.strip() + '\n\n'
        paragraphs = [x.strip() for x in re.split('</?p>', body) if len(x.strip()) > 0]
        out_str += '\n\n'.join(paragraphs)
        out_str += '\n\n'
    return out_str.strip()


def process(row):
    row['sections'] = clean(row['sections'])
    text = create_input(row)
    return {
        'id': row['uuid'],
        'journal': row['article_source'],
        'num_tokens': len(re.split(r'\W+', text)),
        'text': text
    }


if __name__ == '__main__':
    """
    Remove Pubmed from ChemSum
    """

    dataset = load_dataset('griffin/ChemSum')
    concat_dataset = concatenate_datasets([dataset['train'], dataset['validation']])

    concat_dataset = concat_dataset.filter(lambda row: 'pubmed' not in row['article_source'].lower())

    remove_columns = ['uuid', 'abstract', 'source_toks', 'target_toks', 'compression', 'headers', 'sections', 'article_source']
    
    out_data = concat_dataset.map(
        process,
        num_proc=16,
        remove_columns=remove_columns,
    )

    out_data = Dataset.from_list(out_data)

    out_fn = os.path.join(OUT_DIR, 'dataset_hf_v2')
    print(f'Saving {len(out_data)} examples to {out_fn}')
    out_data.save_to_disk(out_fn)
