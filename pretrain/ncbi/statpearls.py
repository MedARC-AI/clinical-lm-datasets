import os
import json
import requests

import regex as re
import ftfy
from bs4 import BeautifulSoup
from datasets import Dataset
from p_tqdm import p_uimap
from trafilatura import fetch_url, extract
from tqdm import tqdm

BASE_URL = 'https://www.ncbi.nlm.nih.gov'
PEARL_URL = os.path.join(BASE_URL, 'books/NBK430685')
MIN_CHAPTER_TOKENS = 50

OUT_DIR = '/weka/home-griffin/clinical_pile/ncbi_bookshelf'
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DATA_DIR = os.path.join(OUT_DIR, 'dataset_hf')


def extract_data(link):
    downloaded = fetch_url(link)
    if downloaded is None:
        print(f'Failed to download {link}')
        return None

    # TODO - Extract elements themselves
    result = extract(downloaded)
    end_of_headers = re.search(r'statpearls \[internet\]\.show details', result, flags=re.IGNORECASE)
    references_match = re.search(r'\nreferences\n', result, flags=re.IGNORECASE)

    assert 'StatPearls [Internet].Show details' in result
    if references_match is None:
        print(result)
        references_match = re.search(r'\ndisclosure:', result, flags=re.IGNORECASE)

    result_trunc = result[end_of_headers.end():references_match.start()].strip()
    result_trunc = clean(result_trunc)

    num_tokens = len(re.split('\W+', result_trunc))
    if num_tokens >= MIN_CHAPTER_TOKENS:
        out_row = {
            'id': link,
            'link': link,
            'title': 'statpearls',
            'num_tokens': num_tokens,
            'text': result_trunc
        }

        return out_row
    else:
        print(result_trunc)
        print(f'Chapter ({link}) too short: {num_tokens} < {MIN_CHAPTER_TOKENS}!')
        return None


def clean(text):
    encoded = ftfy.fix_text(text.encode().decode('unicode_escape', 'ignore'))
    encoded = encoded.replace('Å', '').replace('À', '').replace('Â', '').replace('\ue103', 'f').replace('\ue104', 'fl').replace('\ue09d', 'ft')
    encoded = re.sub(f'\[[\d,\s]+\]', ' ', encoded)
    encoded = re.sub('\s+', ' ', encoded)
    return encoded


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    link_fn = os.path.join(OUT_DIR, 'links.txt')

    # Fetch the HTML content of the webpage
    response = requests.get(PEARL_URL)
    html_content = response.text

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all <a> tags which typically denote links
    links = soup.find_all('a')

    chapter_links = []
    # Extract the href attribute from each <a> tag
    for link in links:
        href = link.get('href')
        if '/books/n/statpearls/article' in href:
            slash_ct = 0
            if BASE_URL.endswith('/'):
                slash_ct += 1
            if href.startswith('/'):
                slash_ct += 1
            assert slash_ct == 1
            chapter_links.append(BASE_URL + href)

    print(f'Saving {len(chapter_links)} links to {link_fn}')
    with open(link_fn, 'w') as fd:
        fd.write('\n'.join(chapter_links))

    out_data = list(filter(None, list(p_uimap(extract_data, chapter_links, num_cpus=16))))
    out_data = Dataset.from_list(out_data)
    print(f'Saving {len(out_data)} chapters to {OUT_DATA_DIR}')
    out_data.save_to_disk(OUT_DATA_DIR)
