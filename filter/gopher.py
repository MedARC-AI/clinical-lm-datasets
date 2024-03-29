import multiprocess
import re
import string
from collections import Counter

import argparse
from datasets import concatenate_datasets, load_dataset, load_from_disk
import json
import numpy as np
from tqdm import tqdm
from p_tqdm import p_uimap
from nltk.tokenize import word_tokenize

STOP_WORDS = ['the', 'be',  'to', 'of', 'and', 'that', 'have', 'with']
SPECIAL_TOKENS = ['[/bib_ref]', '[bib_ref]', '[fig]', '[/fig]', '#', '##', '###']


"""
Table A1 from https://arxiv.org/pdf/2112.11446.pdf
    duplicate line fraction                 0.30
    duplicate paragraph fraction            0.30
    duplicate line character fraction       0.20
    duplicate paragraph character fraction  0.20

    top 2-gram character fraction           0.20
    top 3-gram character fraction           0.18
    top 4-gram character fraction           0.16

    duplicate 5-gram character fraction     0.15
    duplicate 6-gram character fraction     0.14
    duplicate 7-gram character fraction     0.13
    duplicate 8-gram character fraction     0.12
    duplicate 9-gram character fraction     0.11
    duplicate 10-gram character fraction    0.10
"""


def get_n_grams(words, n):
    return [' '.join(words[i : i + n]) for i in range(len(words) - n + 1)]


def find_duplicates(x):
    unique_x = set()
    duplicate_chars = 0
    duplicate_elements = 0
    for element in x:
        if element in unique_x:
            duplicate_chars += len(element)
            duplicate_elements += 1

        else:
            unique_x.add(element)
    return duplicate_elements, duplicate_chars


def find_top_duplicate(x):
    counter = Counter()
    for element in x:
        counter[element] += 1
    top_n_gram = counter.most_common(1)[0]
    return len(top_n_gram[0]) * top_n_gram[1]


def find_all_duplicate(words, n):
    n_words = len(words)
    unique = set()
    repeated_chars, idx = 0, 0
    while idx < n_words - n + 1:
        n_gram = ''.join(words[idx : idx + n])
        if n_gram in unique:
            repeated_chars += len(n_gram)
            idx += n
        else:
            unique.add(n_gram)
            idx += 1
    assert repeated_chars <= len(''.join(words))
    return repeated_chars


class GopherRepetitionFilter:
    def __init__(
        self,
        dup_line_frac: float | None = 0.3,
        dup_para_frac: float | None = 0.3,
        dup_line_char_frac: float | None = 0.2,
        dup_para_char_frac: float | None = 0.2,
        top_n_grams = ((2, 0.2), (3, 0.18), (4, 0.16)),
        dup_n_grams = ((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.10)),
    ):
        """

        Args:
            dup_line_frac:
            dup_para_frac:
            dup_line_char_frac:
            dup_para_char_frac:
            top_n_grams:
            dup_n_grams:
]        """
        self.dup_line_frac = dup_line_frac
        self.dup_para_frac = dup_para_frac
        self.dup_line_char_frac = dup_line_char_frac
        self.dup_para_char_frac = dup_para_char_frac
        self.top_n_grams = top_n_grams
        self.dup_n_grams = dup_n_grams
        self.paragraph_exp = re.compile(r'\n{2,}')

    def filter(self, text, words):
        paragraphs = self.paragraph_exp.split(text.strip())
        paragraphs_duplicates, char_duplicates = find_duplicates(paragraphs)
        if self.dup_para_frac and paragraphs_duplicates / len(paragraphs) > self.dup_para_frac:
            return False, 'dup_para_frac'
        if self.dup_para_char_frac and char_duplicates / len(text) > self.dup_para_char_frac:
            return False, 'dup_para_char_frac'

        lines = [x.strip() for x in text.splitlines() if len(x.strip()) > 0]
        line_duplicates, char_duplicates = find_duplicates(lines)
        if self.dup_line_frac and line_duplicates / len(lines) > self.dup_line_frac:
            return False, 'dup_line_frac'
        if self.dup_line_char_frac and char_duplicates / len(text) > self.dup_line_char_frac:
            return False, 'dup_line_char_frac'

        # words = word_tokenize(text, language='english')

        for n, n_frac in self.top_n_grams:
            n_grams = get_n_grams(words, n)
            if not n_grams:
                continue
            top_char_length = find_top_duplicate(n_grams)
            if top_char_length / len(text) > n_frac:
                return False, f'top_{n}_gram'

        for n, n_frac in self.dup_n_grams:
            n_duplicates_char = find_all_duplicate(words, n)
            if n_duplicates_char / len(text) > n_frac:
                return False, f'duplicated_{n}_n_grams'

        return True, ''


class GopherQualityFilter:
    def __init__(
        self,
        min_doc_words: int | None = 50,
        max_doc_words: int | None = 100000,
        min_avg_word_length: int | None = 3,
        max_avg_word_length: int | None = 10,
        max_symbol_word_ratio: float | None = 0.2,  # Set to None since we use "#" for headers
        max_bullet_lines_ratio: float | None = 0.9,
        max_ellipsis_lines_ratio: float | None = 0.3,
        max_non_alpha_words_ratio: float | None = 0.8,
        min_stop_words: int | None = 2,
        stop_words=None,
    ):
        """
        Filter to apply Gopher's quality heuristic rules.
        Reference: https://arxiv.org/pdf/2112.11446.pdf

        Args:
            min_doc_words:
            max_doc_words:
            min_avg_word_length:
            max_avg_word_length:
            max_symbol_word_ratio:
            max_bullet_lines_ratio:
            max_ellipsis_lines_ratio:
            max_non_alpha_words_ratio:
            min_stop_words:
            stop_words:
        """
        self.min_doc_words = min_doc_words
        self.max_doc_words = max_doc_words
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.max_symbol_word_ratio = max_symbol_word_ratio
        self.max_bullet_lines_ratio = max_bullet_lines_ratio
        self.max_ellipsis_lines_ratio = max_ellipsis_lines_ratio
        self.max_non_alpha_words_ratio = max_non_alpha_words_ratio
        self.min_stop_words = min_stop_words
        self.stop_words = set(STOP_WORDS if stop_words is None else stop_words)

    def filter(self, text, words):
        """

        Args:
            doc: Applies the heuristics rules to decide if a document should be REMOVED


        Returns: False if sample.text does not pass any of the the heuristic tests

        """

        # words < min_doc_words or words > max_doc_words
        n_words = len([w for w in words if w not in string.punctuation])
        if self.min_doc_words and n_words < self.min_doc_words:
            return False, 'gopher_short_doc'
        if self.max_doc_words and n_words > self.max_doc_words:
            return False, 'gopher_long_doc'

        # mean word length is outside the range of 3 to 10 characters
        avg_n_words = np.mean([len(w) for w in words if w not in string.punctuation])
        if self.min_avg_word_length and avg_n_words < self.min_avg_word_length:
            return False, 'gopher_below_avg_threshold'
        if self.max_avg_word_length and avg_n_words > self.max_avg_word_length:
            return False, 'gopher_above_avg_threshold'

        # symbol-to-word ratio greater than 0.1 for either the hash symbol or the ellipsis
        if self.max_symbol_word_ratio and text.count('#') / n_words > self.max_symbol_word_ratio:
            return False, 'gopher_too_many_hashes'
        if self.max_symbol_word_ratio and (text.count('...') + text.count('…')) / n_words > self.max_symbol_word_ratio:
            return False, 'gopher_too_many_ellipsis'

        # any document with more than 90 % of lines starting with a bullet point,
        # or more than 30 % ending with an ellipsis.
        lines = text.splitlines()
        if (
            self.max_bullet_lines_ratio
            and sum(s.lstrip().startswith('•') or s.lstrip().startswith('-') for s in lines) / len(lines)
            > self.max_bullet_lines_ratio
        ):
            return False, 'gopher_too_many_bullets'
        if (
            self.max_ellipsis_lines_ratio
            and sum(s.rstrip().endswith('...') or s.rstrip().endswith('…') for s in lines) / len(lines)
            > self.max_ellipsis_lines_ratio
        ):
            return False, 'gopher_too_many_end_ellipsis'

        # that 80 % of words in a document contain at least one alphabetic character
        if (
            self.max_non_alpha_words_ratio
            and sum([any((c.isalpha() for c in w)) for w in words]) / n_words < self.max_non_alpha_words_ratio
        ):
            return False, 'gopher_below_alpha_threshold'

        # stop word filter
        if self.min_stop_words and sum(w in self.stop_words for w in words) < self.min_stop_words:
            return False, 'gopher_enough_stop_words'

        return True, ''


def gopher_removal_reason(doc, rep_filter, quality_filter):
    words = word_tokenize(doc['text'], language='english')
    is_quality, reason = rep_filter.filter(doc['text'], words)

    if not is_quality:
        # print(doc['text'])
        # print(reason)
        # print('\n' + '*' * 50 + '\n')
        return reason  # {'text': doc['text'], 'gopher_reason': reason}

    is_quality, reason = quality_filter.filter(doc['text'], words)

    if not is_quality:
        return reason
        # return {'text': doc['text'], 'gopher_reason': reason}

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running LLM-based quality filter on paragraphs for different datasets.')

    parser.add_argument('--pile_path', default='/weka/home-griffin/clinical_pile/v1/dedup/sentence/dataset.jsonl')
    parser.add_argument('--excluded_sources', default='code|gutenberg_books')

    args = parser.parse_args()

    excluded_sources = set(args.excluded_sources.split('|'))

    rep_filter = GopherRepetitionFilter()

    # We use "#" for document structure. Don't want to remove documents with these symbols
    # Max doc words should be none or else it will remove all books.
    quality_filter = GopherQualityFilter(max_symbol_word_ratio=None, max_doc_words=None)

    from collections import defaultdict
    removal_reasons = defaultdict(int)
    num_saved = 0
    num_removed = 0

    dataset = load_dataset('json', data_files=args.pile_path, split='train')
    cols = list(dataset.features)
    print(len(dataset))

    def gopher_filters(obj, rep_filter, quality_filter):
        # line = line.strip()
        # obj = json.loads(line)

        if obj['source'] in args.excluded_sources:
            is_keep = True
            gopher_reason = None
        else:
            gopher_reason = gopher_removal_reason(obj, rep_filter, quality_filter)
            is_keep = gopher_reason is None

        return is_keep

    n = len(dataset)
    dataset = dataset.filter(
        lambda row: gopher_filters(row, rep_filter, quality_filter), num_proc=multiprocess.cpu_count()
    )

    filt_n = len(dataset)
    print(f'{n} -> {filt_n} after running through Gopher rules...')

    dataset.save_to_disk('/weka/home-griffin/clinical_pile/v1/dataset_hf_clean')
