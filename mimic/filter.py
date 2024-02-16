import os
import regex as re
from typing import List
from dataclasses import dataclass

import argparse
import pandas as pd
import spacy
from datasets import load_from_disk

from mimic.combine import FILTERED_NOTES_DIR
from mimic.constants import MIMIC_DIR
from utils import sent_tokenize_rules


_DEFAULT_MIN_NOTE_TOKENS = 50


@dataclass
class Section:
    header: str
    sentences: List[str]


def should_include_report(section, nlp):
    if section.header is not None:
        if 'technique' in section.header.lower():
            return False

        if 'comparison' in section.header.lower():
            return False

        if 'impression' in section.header.lower():
            return True

        if 'finding' in section.header.lower():
            return True

    toks = re.split('\s+', ' '.join(section.sentences))
    if len(toks) <= 3:
        return False

    # Number of tokens with punct or number
    num_non_alpha = 0
    for tok in toks:
        if re.search(r'[0-9]', tok) is not None or re.match(r'\W', tok) is not None:
            num_non_alpha += 1

    frac_non_alpha = num_non_alpha / len(toks)
    if frac_non_alpha >= 0.5:
        return False

    # Add an entity filter?
    ents = nlp(' '.join(section.sentences)).ents
    if len(ents) == 0:
        return False

    return True

def should_include_dsum(section, nlp):
    if section.header is not None:
        if 'course' in section.header.lower():
            return True

        if 'date' in section.header.lower():
            return False
        
        if 'signature' in section.header.lower():
            return False

        if 'service' in section.header.lower():
            return False

        if 'dictated' in section.header.lower():
            return False
    
    toks = re.split('\s+', ' '.join(section.sentences))
    if len(toks) <= 3:
        return False

    # Number of tokens with punct or number
    num_non_alpha = 0
    for tok in toks:
        if re.search(r'[0-9]', tok) is not None or re.match(r'\W', tok) is not None:
            num_non_alpha += 1

    frac_non_alpha = num_non_alpha / len(toks)
    if frac_non_alpha >= 0.5:
        return False

    # Add an entity filter?
    ents = nlp(' '.join(section.sentences)).ents
    if len(ents) == 0:
        return False

    return True


def filtered_dsum(sections, nlp):
    output_lines = []

    for section in sections:
        if should_include_dsum(section, nlp):
            if section.header is not None:
                output_lines.append('## ' + section.header.upper())
            for sent in section.sentences:
                # TODO: Do an ent filter here too?
                output_lines.append(sent)
            output_lines.append('')
    
    return '\n'.join(output_lines)


def filtered_report(sections, nlp):
    output_lines = []
    for section in sections:
        if should_include_report(section, nlp):
            if section.header is not None:
                output_lines.append('## ' + section.header.upper())
            for sent in section.sentences:
                output_lines.append(sent)
            output_lines.append('')
    return '\n'.join(output_lines)


def parse_lines(text, max_header_toks=5):
    toks = text.split(' ')
    cand_header_toks = toks[:min(len(toks), max_header_toks)]
    header_idx = None
    for cand_idx in range(len(cand_header_toks)):
        if cand_header_toks[cand_idx].endswith(':'):
            header_idx = cand_idx
            break

    if header_idx is None:
        return {'text': text}

    return {'text': ' '.join(toks[header_idx + 1:]), 'header': ' '.join(toks[:header_idx + 1])}


def construct_sections(parsed):
    sections = []
    curr_header = None
    curr_sents = []
    for line in parsed:
        text = line['text'].strip()
        if 'header' in line:
            if len(curr_sents) > 0:
                sections.append(Section(header=curr_header, sentences=curr_sents))
            curr_sents = []
            curr_header = line['header']
        if len(text) > 0:
            curr_sents.append(text)
    if len(curr_sents) > 0:
        sections.append(Section(header=curr_header, sentences=curr_sents))

    return sections


def filter_text(args, row, nlp):
    text = re.sub(r'[ _]+', ' ', row['text'])
    nt = row['note_type']
    text_segments = sent_tokenize_rules(text)
    text_parsed = list(map(parse_lines, text_segments))
    sections = construct_sections(text_parsed)
    if nt.lower() == 'discharge summary':
        filtered = filtered_dsum(sections, nlp)
    elif nt.lower() == 'radiology':
        filtered = filtered_report(sections, nlp)
    else:
        raise Exception('Unexpected note type -> {}')

    num_tokens = len(re.split('\W+', filtered))
    if num_tokens < args.min_note_tokens:
        return {'text': None, 'num_tokens': None}

    return {'text': filtered, 'num_tokens': num_tokens}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to section filter combined MIMIC-III and MIMIC-IV Discharge Summaries and Radiology Reports.')
    
    parser.add_argument('--min_note_tokens', default=_DEFAULT_MIN_NOTE_TOKENS, type=int)

    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()

    data_dir = os.path.join(MIMIC_DIR, 'mini_hf') if args.debug else FILTERED_NOTES_DIR
    print(f'Loading dataset from {data_dir}...')
    notes = load_from_disk(data_dir)
    
    print('Loading SciSpacy')
    nlp = spacy.load('en_core_sci_sm')

    notes = notes.map(
        lambda row: filter_text(args, row, nlp),
        num_proc=32
    )

    notes = notes.filter(lambda row: row['text'] is not None)

    if not args.debug:
        OUT_DIR = os.path.join(MIMIC_DIR, 'dataset_hf')
        print(f'Saving {len(notes)} examples to {OUT_DIR}')
        notes.save_to_disk(OUT_DIR)
