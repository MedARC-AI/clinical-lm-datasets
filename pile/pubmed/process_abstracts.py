"""
This script implements the pre-processing of PubMed full-text articles and abstracts. 
Including: Filtering, formatting, cleaning. 
"""

import argparse
import json
import numpy as np
import tqdm.auto as tqdm
import json
from itertools import groupby
import re
import os
from langdetect import detect
import jsonlines

from load import *

KEEP_HEADER = False         # Keep article header (content before title/abstract/first section header)?
KEEP_FIGURE_CONTENT = True  # Keep figure content and wrap in [fig] tokens?
KEEP_TABLE_CONTENT = True   # Keep table content and wrap in [table] tokens? 
KEEP_BIBLIOGRAPHY = False   # Keep bibligraphy entries and wrap in [bib] tokens?

SPECIAL_TOKENS = [
    '[bib_ref]', '[/bib_ref]',  # In-text author references
    '[fig_ref]', '[/fig_ref]',  # In-text figure references
    '[formula]', '[/formula]'   # In-text formulae
    ]
if KEEP_FIGURE_CONTENT:
    SPECIAL_TOKENS += ['[fig]', '[/fig]']
if KEEP_TABLE_CONTENT:
    SPECIAL_TOKENS += ['[table]', '[/table]']
if KEEP_BIBLIOGRAPHY: 
    SPECIAL_TOKENS += ['[bib]', '[/bib]']


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


def format_bib(record, bib_id, max_length=12):
    ''' 
    Format in-text bibliography reference into (paper title, main author last name).
    Truncates bibliography title to max_length words if needed. 
    '''
    article = record['content']['text']
    annotations = record['content']['annotations']
    try:
        # Find bib entry 
        for bib_entry in json.loads(annotations['bibentry']):
            if bib_entry['attributes']['id'] == bib_id:
                entry_start = int(bib_entry['start'])
                entry_end = int(bib_entry['end'])
                break

        # Find title 
        for bib_title in json.loads(annotations['bibtitle']):
            if bib_title['start'] >= entry_start and bib_title['end'] <= entry_end:
                bib_title_str = article[int(bib_title['start']):int(bib_title['end'])]
                break

        # If no title found, skip reference
        if not bib_title_str:
            return None

        # Find main author's last name
        for bib_author in json.loads(annotations['bibauthorlastname']):
            if bib_author['start'] >= entry_start and bib_author['end'] <= entry_end:
                bib_author_name = article[int(bib_author['start']):int(bib_author['end'])]
                break
        if not bib_author_name:
            return None
    except:
        return None

    # Format bibliography reference
    split = bib_title_str.split()
    if len(split) > max_length:
        bib_title_str = ' '.join(split[:max_length])+'...'
    bib_str = f"{bib_title_str}, {bib_author_name}"
    return bib_str


def format_fig(record, fig_id, max_length=12):
    '''
    Format figure reference into `Fig [ID]: [summarized figure caption].`
    Truncates figure caption to max_length words if needed. 
    '''
    article = record['content']['text']
    annotations = record['content']['annotations']
    try:
        # Find figure entry
        for fig in json.loads(annotations['figure']):
            if fig['attributes']['id'] == fig_id:
                fig_start = int(fig['start'])
                fig_end = int(fig['end'])
                break


        # Find figure caption
        fig_caption_start, fig_caption_end = None, None
        for fig_caption in json.loads(annotations['figurecaption']):
            if fig_caption['start'] >= fig_start and fig_caption['end'] <= fig_end:
                fig_caption_start = int(fig_caption['start'])
                fig_caption_end = int(fig_caption['end'])
                break

        # If no caption found, skip
        if not fig_caption_start:
            return None, None

        # Format prefix 
        prefix = article[fig_start:fig_caption_start].split('\n')[1]
        fig_name = re.sub(r'[:()]', '', prefix.replace(' .', ' '))
        fig_name = fig_name.replace('Fig.', 'Figure')
        fig_name = fig_name.replace('Tab.', 'Table')
        fig_name = fig_name.replace(' Figure', ', Figure')
        fig_name = fig_name.strip()

        # Format caption
        caption = article[fig_caption_start:fig_caption_end].replace(prefix, '').strip()
        if max_length:
            caption = summarize_caption(caption, max_length)
        if caption.split()[0].isdigit():
            fig_name += ' '+caption.split()[0]
            caption = ' '.join(caption.split()[1:])
        if fig_name != '':
            fig_name += ': '
        while caption.startswith('.') or caption.startswith(',') or caption.startswith(')'):
            caption = caption[1:].strip()
        return fig_name, caption
    except:
        return None


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
        annotations = json.loads(annotations)

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
    if not KEEP_HEADER: 
        try:
            start = None
            abstract = record['content']['annotations']['abstract']
            if abstract:
                abstract_start = int(json.loads(abstract)[0]['start'])
                if abstract_start:
                    start = abstract_start
            section_headers = json.loads(record['content']['annotations']['sectionheader'])
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


def process_abstracts(abstracts, save_path, start=None, end=None):
    ''' 
    Processing for PubMed abstracts.
    '''
    if os.path.exists(save_path):
        print(f'File {save_path} already exists. Do you want to overwrite it? [y/n]')
        if input().lower() == 'y':
            os.remove(save_path)
    total = 0
    count = 0
    non_english = 0
    duplicates = 0
    skipped = 0
    with open(save_path, 'a') as f_out:
        for record in tqdm(abstracts):
            if start and total <= start:
                continue
            if end and total > end:
                break
            total += 1

            try:
                text = record.get('text')
                if not text:
                    skipped += 1
                    continue

                # Filter non-english abstracts
                language = detect_lang(text)
                if language != 'en':
                    non_english += 1
                    continue
                
                # Prepend title if given
                title = record.get('title')
                if title:
                    text = '# ' + title + '\n' + text

                # Cleaning up
                text = remove_urls(text)
                text = remove_references(text)

                record['text'] = text
                f_out.write(json.dumps(record) + '\n')
                count += 1
            except: 
                skipped += 1
                continue

    print(f'Finished processing {count} out of {total} articles\
          \nRemoved {non_english} non-English articles.\
          \nSkipped {skipped} articles leading to errors.')


def deduplicate(abstracts, s2orc_path):
    '''
    Remove all abstracts for which we already have a full-text version.
    '''
    # Get all corpus IDs in s2orc_path
    corpus_ids = set()
    print(f'Collecting corpus IDs from {s2orc_path}')
    for line in tqdm(open(s2orc_path, 'r')):
        corpus_ids.add(int(re.search(r'corpusid": (\d+)', line).group(1)))
    print(f'Loaded {len(corpus_ids)} uniqued corpus ids from {s2orc_path}')

    # Remove all abstracts with corpus IDs in s2orc_path
    print(f'\nRemoving all abstracts with full-text versions in {s2orc_path}.\n')
    removed = 0
    deduped = []
    for record in tqdm(abstracts):
        corpus_id = record.get('corpusid')
        assert type(corpus_id) == int
        if corpus_id and corpus_id in corpus_ids:
            removed += 1
            continue
        deduped.append(record)
    print(f'Removed {removed} abstracts with full-text versions in {s2orc_path}.')

    return record


def self_deduplicate(abstracts_path):
    deduped = []
    removed = 0
    seen_cids = set()
    with open(abstracts_path, 'r') as f_in:
        for line in tqdm(f_in):
            record = json.loads(line)
            corpus_id = record.get('corpusid')
            assert type(corpus_id) == int
            if corpus_id in seen_cids:
                removed += 1
                continue

            seen_cids.add(corpus_id)
            deduped.append(record)
    print(f'Removed {removed} with the exact same S2 Corpus ID')
    return deduped


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path", type=str,
        default='/weka/home-griffin/clinical_pile/pubmed/abstracts/abstracts-PubMed_metadata.jsonl',
        help="Path to jsonl file to process."
    )
    parser.add_argument(
        "--dedup_path", type=str,
        default='/weka/home-griffin/clinical_pile/pubmed/abstracts/abstracts-PubMed_dedup.jsonl',
        help="Path to save processed deduped jsonl file."
    )
    parser.add_argument(
        "--processed_path", type=str,
        default='/weka/home-griffin/clinical_pile/pubmed/abstracts/abstracts-PubMed_processed.jsonl',
        help="Path to save processed fully processed jsonl file."
    )
    parser.add_argument(
        '--s2orc_path', default='/weka/home-griffin/clinical_pile/pubmed/s2orc/s2orc-PubMed_metadata.jsonl',
        help='Path to S2orc data for de-duplication.'
    )
    parser.add_argument(
        "--deduplicate",
        action='store_true',
        help="If passed as argument, remove all abstracts for which we already have a full-text version.")

    args = parser.parse_args()

    filtered_abstracts = self_deduplicate(args.source_path)
    filtered_abstracts = deduplicate(filtered_abstracts, args.s2orc_path)
    process_abstracts(filtered_abstracts, args.processed_path)
