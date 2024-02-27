"""
This script implements the pre-processing of PubMed full-text articles and abstracts. 
Including: Filtering, formatting, cleaning. 
"""

import argparse
from datasets import load_dataset
import orjson  # For faster de-serialization (loads)
import json
import numpy as np
import tqdm.auto as tqdm
from itertools import groupby
import re
import string
import os
from langdetect import detect
from p_tqdm import p_uimap

from load import *

KEEP_HEADER = False  # Keep article header (content before title/abstract/first section header)?
KEEP_BIBLIOGRAPHY = False  # Keep bibligraphy entries and wrap in [bib] tokens?


MAIN_SECTION_HEADERS = [
    'Abstract', 'Introduction', 'Background', 'Related',
    'Method', 'Material', 'Result', 'Analysis', 'Discussion',
    'Conclusion', 'Contribution', 'Statement', 'Declaration', 
    'Strength', 'Limitation', 'Future research', 'Funding',
    'Disclosure', 'Acknowledgment', 'Ethical', 
    'Tables', 'Figures', 'Appendix'
]


def detect_lang(text, sample_size=2000): 
    '''
    Helper: Detect language of a given text.
    '''
    try:
        sample = text if len(text) < sample_size else text[:sample_size]
        language = detect(sample)
    except:
        language = 'unknown'
    return language


def remove_urls(text):
    '''
    Helper: remove URLs from text.
    '''
    return re.sub(
        r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b', '', 
        text, flags=re.MULTILINE)


def remove_references(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)
    return text


def summarize_caption(caption, max_length):
    ''' 
    Helper: summarize figure caption to max_length words.  
    '''
    # Truncate first sentence if > 20 characters
    if len(caption) > 20:
        caption = re.split(r'[.;:()]', caption)[0]
    # Truncate to max_length words if needed
    split = caption.split()
    if len(split) > max_length:
        caption = ' '.join(split[:max_length])+'...'
    return caption


def is_main_section_header(section):
    '''
    Helper: check if a section header is a usual main section header.
    '''
    if len(section.split(' ')) > 3: 
        return False
    for header in MAIN_SECTION_HEADERS:
        if header.lower() in section.lower():
            return True
    return False


def parse_article(record):
    '''
    Creates an array of annotation types for each character in the article.
    This array is then used to format the article using the `format_article` function.
    '''
    article = record['content']['text']
    if not article:
        return None
    reflect_array = np.array(['T' for _ in range(len(article))], dtype=object)
    parsing_dict = {
        'authorfirstname': 'RM',
        'authorlastname': 'RM',
        'authoraffiliation': 'RM',
        'bibentry': 'BIB',
        'formula': 'FML',
        'sectionheader': 'SEC',
        'bibref': None,
        'figureref': None,
        'tableref': None,
        'figure': None
    }

    # Parse each annotation type
    for annot_type, token in parsing_dict.items():
        annotations = record['content']['annotations'][annot_type]
        if not annotations:
            continue
        annotations = orjson.loads(annotations)

        # Remove title duplicates
        if annot_type == 'title':
            annotations = [annotations[0]]

        for annotation in annotations:
            start = int(annotation["start"])
            end = int(annotation["end"])
            try:
                # In-text references (skip unidentified ones!)
                if annot_type in ['bibref', 'figureref', 'tableref']:

                    # Fix recurrent parsing error
                    if '(' in article[start-3:start]:
                        start = start-3+article[start-3:start].index('(')
                    if ')' in article[end:end+3]:
                        end = end+article[end:end+3].index(')')+1

                    if 'attributes' in annotation.keys():
                        ref_id = annotation['attributes']['ref_id']
                        reflect_array[start:end] = ref_id
                    else: 
                        reflect_array[start:end] = 'b?'
                elif annot_type == 'figure':
                    fig_id = annotation['attributes']['id']
                    fig_id = fig_id.split('_')[0].upper()+'_'+fig_id.split('_')[1] 
                    reflect_array[start:end] = fig_id
                else:
                    reflect_array[start:end] = token
            except:
                pass

    # Remove article header (before title/abstract/first section header)
    try:
        start = None
        abstract = record['content']['annotations']['abstract']
        if abstract:
            abstract_start = int(orjson(abstract)[0]['start'])
            if abstract_start:
                start = abstract_start
        section_headers = orjson.loads(record['content']['annotations']['sectionheader'])
        if section_headers:
            intro_start = min([int(s['start']) for s in section_headers])
            if not start or intro_start < start:
                start = intro_start
        if start:
            idx_T = np.where(reflect_array == 'T')[0]
            idx_before_abstract = idx_T[idx_T < start]
            reflect_array[idx_before_abstract] = 'P'
    except:
        pass
    return reflect_array


def format_article(record):
    '''
    Full-text article formatting using S2ORC annotations.
    '''
    start = 0
    formatted_figs = {}
    # formatted_bibs = {}
    added_figures = []
    article = record['content']['text']
    text = ''

    # Parse article into array of annotation types
    reflect_array = parse_article(record)

    # Group sections by annotation type
    split_array = [list(group) for _, group in groupby(reflect_array)]
    at_figures = False
    for subarray in split_array:
        end = start + len(subarray)
        annot_type = subarray[0]
        part = article[start:end]

        # Format whitespace and bullet points
        part = part.strip()
        part = part.replace('•', '- ')
        try: 

            # Skip empty sections 
            if part == '':
                start += len(subarray)
                continue 

            # Keep abstract & main body (skip all text after figures)
            elif annot_type == 'T' and not at_figures:
                text += part

            # Format section headers (## for sections, ### for subsections, capitalise first letter)
            elif annot_type == 'SEC':
                part = part[0].upper() + part[1:].lower()
                if is_main_section_header(part):
                    text += '\n## ' + part.strip(string.punctuation).strip().upper() + '\n'
                else:
                    text += '\n### ' + part.strip(string.punctuation).strip().upper() + '\n'

            # Wrap in-text figures/table refs in [fig_ref] tokens + summarize caption
            elif 'fig_' in annot_type or 'tab_' in annot_type:
                text += ' '

            # Wrap in-text author/bib references in [bib_ref] tokens + summarize caption
            elif 'b' in annot_type:
                # Skip unidentified references
                if annot_type == 'b?':
                    text += ' '
                    start += len(subarray)
                    continue


            # Keep figure/table content wrapped in [fig]/[table] tokens
            elif ('FIG_' in annot_type) or ('TAB_' in annot_type):
                at_figures = True

            # Wrap formulae in [formula] tokens
            elif annot_type == 'FML':
                text += '\n[formula] ' + part + ' [/formula]\n'

            # Advance along the article
            start += len(subarray)

        except:
            # If there's any error in a part, just skip it
            start += len(subarray)
            continue

    # Further formatting
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\n# ', '\n\n# ', text)
    text = re.sub(r'\n## ', '\n\n## ', text)
    text = re.sub(r'\n### ', '\n\n### ', text)
    text = re.sub(r' +', ' ', text)

    return text


def process_s2orc(source_path, save_path):
    '''
    Pre-processing for full-text PubMed articles in S2ORC format.
    '''
    print(f'\nPre-processing PubMed articles in {source_path}.\n')
    if os.path.exists(save_path):
        print(f'File {save_path} already exists. Do you want to overwrite it? [y/n]')
        if input().lower() == 'y':
            os.remove(save_path)
    
    count = 0
    skipped = 0
    # with open(source_path, 'r') as f_in, open(save_path, 'a') as f_out:
    # for line in tqdm(f_in):
    f_in = open(source_path, 'r')

    full_data = list(p_uimap(process_line, f_in))
    # full_data = list(map(process_line, f_in))
    filtered_data = list(filter(None, full_data))

    total = len(full_data)
    count = len(filtered_data)
    skipped = total - count

    print(f'Finished processing {count} out of {total} articles\
          \nSkipped {skipped} articles leading to errors. ')
 
    print(f'Saving to {save_path}')
    with open(save_path, 'w') as f_out:
        for record in tqdm(filtered_data):
            f_out.write(json.dumps(record) + '\n')


def process_line(line):
    # Filter out invalid entries
    record = orjson.loads(line)
    content = record.get('content')
    if not content:
        skipped += 1
        return None
    text = content.get('text')
    if not text:
        return None

    # Filter non-english articles
    language = detect_lang(text)
    if language != 'en':
        return None

    # Format article
    text = format_article(record)
    if not text:
        return None

    # Prepend "# {title}" if given
    title = record.get('title')
    if title:
        text = '# ' + title + '\n\n' + text

    text = text.strip()

    pmid = str(record['externalids'].get('PubMed', ''))
    pmcid = str(record['externalids'].get('PubMedCentral', ''))
    doi = str(record['externalids'].get('DOI', ''))
    id = []
    if pmid is not None:
        id.append('pmid-' + pmid)
    if pmcid is not None:
        id.append('pmcid-' + pmcid)
    id = '_'.join(id)

    obj = {
        'id': id,
        'pmid': pmid,
        'pmcid': pmcid,
        'doi': doi,
        's2_corpusid': str(record['corpusid']),
        'text': text,
        'isopenaccess': record['isopenaccess'],
        'journal': str(record['journal']['name']),
    }

    return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path", type=str, default='/weka/home-griffin/clinical_pile/pubmed/s2orc/s2orc-PubMed_metadata.jsonl',
        help="Path to jsonl file to process.")
    parser.add_argument(
        "--save_path", type=str,
        default='/weka/home-griffin/clinical_pile/pubmed/s2orc/s2orc-PubMed_processed',
        help="Path to save processed jsonl file.")
    args = parser.parse_args()

    json_out = args.save_path + '.jsonl'
    hf_out = args.save_path + '_hf'
    cids_out = args.save_path + '_corpusids.txt'
    process_s2orc(args.source_path, json_out)

    hf_dataset = load_dataset('json', data_files=json_out, split='train')

    hf_dataset = hf_dataset.map(
        lambda row: {'num_tokens': len(re.split(r'\W+', row['text']))},
        num_proc=64
    )

    print(f'Saving {len(hf_dataset)} examples to {hf_out}')
    hf_dataset.save_to_disk(hf_out)

    # Used when getting PeS2o data to remove PubMed articles in S2
    corpus_ids = list(sorted(list(set(hf_dataset['s2_corpusid']))))
    with open(cids_out, 'w') as fd:
        fd.write('\n'.join(corpus_ids))
