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
from p_tqdm import p_uimap

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
                    text += '\n## ' + part + '\n'
                else:
                    text += '\n### ' + part + '\n'

            # Wrap entries in special tokens [bib] (only if KEEP_BIBLIOGRAPHY)
            elif annot_type == 'BIB' and KEEP_BIBLIOGRAPHY: 
                text += ' [bib] ' + part + ' [/bib]\n'

            # Wrap in-text figures/table refs in [fig_ref] tokens + summarize caption
            elif 'fig_' in annot_type or 'tab_' in annot_type:
                if annot_type in formatted_figs:
                    fig_str = formatted_figs[annot_type]
                else:
                    fig_name, caption = format_fig(record, annot_type)
                    fig_str = fig_name + caption
                    formatted_figs[annot_type] = fig_str
                if fig_str:
                    text += ' [fig_ref] ' + fig_str + ' [/fig_ref] '

            # Wrap in-text author/bib references in [bib_ref] tokens + summarize caption
            elif 'b' in annot_type:
                # Skip unidentified references
                if annot_type == 'b?':
                    text += ' '
                    start += len(subarray)
                    continue

                # Format identified references
                # I removed the in-line citations
                # if annot_type in formatted_bibs:
                #     bib_str = formatted_bibs[annot_type]
                # else:
                #     bib_str = format_bib(record, annot_type)
                #     formatted_bibs[annot_type] = bib_str
                # if bib_str:
                #     text += ' [bib_ref] ' + bib_str + ' [/bib_ref] '

            # Keep figure/table content wrapped in [fig]/[table] tokens
            elif ('FIG_' in annot_type) or ('TAB_' in annot_type):
                at_figures = True
                fig_id = annot_type.split('_')[0].lower()+'_'+annot_type.split('_')[1]
                fig_name, caption = format_fig(record, fig_id, max_length=None)
                if fig_name and caption:
                    fig_str = fig_name + caption
                    # Check the figure hasn't already been added
                    added = any([re.sub(r'[:,()]', '', fig.strip()) in fig_name.lower() for fig in added_figures])
                    if 'continued' not in fig_str.lower() and not added:
                        added_figures += [fig_name.lower()]
                        tags = ['[fig]','[/fig]'] if 'FIG_' in annot_type else ['[table]','[/table]']
                        text += '\n' + tags[0] + ' ' + fig_str + ' ' + tags[1] + '\n'

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
    text = re.sub(r'\[/fig_ref\] \.', '[/fig_ref].', text)
    text = re.sub(r'\[/bib_ref\] \.', '[/bib_ref].', text)

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
    record = json.loads(line)
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

    # Save article
    record.update({'text': text})
    record.pop('content')
    return record



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path", type=str, default='/weka/home-griffin/clinical_pile/pubmed/s2orc/s2orc-PubMed_metadata.jsonl',
        help="Path to jsonl file to process.")
    parser.add_argument(
        "--save_path", type=str,
        default='/weka/home-griffin/clinical_pile/pubmed/s2orc/s2orc-PubMed_processed_v2.jsonl',
        help="Path to save processed jsonl file.")
    args = parser.parse_args()

    process_s2orc(args.source_path, args.save_path)
