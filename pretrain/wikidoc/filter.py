import os
import requests
import regex as re
from pathlib import Path

import argparse
import ftfy
from bs4 import BeautifulSoup
from datasets import Dataset, load_from_disk
from tqdm import tqdm


BASE_URL = 'https://www.wikidoc.org'
MIN_SECTION_TOKENS = 25
OUT_DIR = '/weka/home-griffin/clinical_pile/wikidoc'
os.makedirs(OUT_DIR, exist_ok=True)


def clean(text):
    encoded = ftfy.fix_text(text.encode().decode('unicode_escape', 'ignore'))
    encoded = encoded.replace('Å', '').replace('À', '').replace('Â', '').replace('\ue103', 'f').replace('\ue104', 'fl').replace('\ue09d', 'ft')
    encoded = re.sub(f'\[[\d,\s]+\]', ' ', encoded)
    encoded = re.sub('[ \t]+', ' ', encoded)
    return encoded


def get_text_tags(tag):
    outputs = []
    if tag.name in {'h2', 'p'}:
        outputs.append(tag)
    for child in tag.children:
        if child.name is not None:
            outputs += get_text_tags(child)
    return outputs


def remove_non_alpha_numeric(url):
    pattern = re.compile('[^a-zA-Z0-9]')
    return pattern.sub('', url.strip().lower()).strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract HTML from WikiDoc.')
    # parser.add_argument('--do_not_filter_sources', default='wikidoc', type='str')  # "|" comma delimited
    parser.add_argument('-remove_guideline_urls', default=False, action='store_true')  # From Meditron

    args = parser.parse_args()

    ## Many pages with "There is currently no text in this page."
    script_dir = Path(__file__).resolve().parent
    filename = 'chapter_links.txt'
    file_path = script_dir / filename

    with open(file_path, 'r') as file:
        chapter_links = [x.strip() for x in file.readlines() if len(x) > 0]

    wikidoc_guideline_urls = set()
    if args.remove_guideline_urls:
        GUIDELINES_DIR = '/weka/home-griffin/clinical_pile/guidelines/dataset_hf'
        print(f'Loading guidelines from {GUIDELINES_DIR}')
        guidelines = load_from_disk(GUIDELINES_DIR)

        wikidoc_guideline_urls = set()
        for g in guidelines:
            url = g['url']
            if 'wikidoc' not in url:
                continue
            assert 'https://www.wikidoc.org/index.php/' in url
            url = url.replace('https://www.wikidoc.org/index.php/', '')
            url_clean = remove_non_alpha(url)
            if len(url_clean) > 0:
                wikidoc_guideline_urls.add(url_clean)

        prev_n = len(chapter_links)
        print(chapter_links[0])
        print(remove_non_alpha(chapter_links[0]))
        chapter_links = [p for p in chapter_links if remove_non_alpha(p.replace('/index.php/', '').strip()) not in wikidoc_guideline_urls]
        new_n = len(chapter_links)
        print(f'Removed {prev_n - new_n} chapters which are in Meditron HuggingFace dataset.')
    
    dataset = []
    seen = set()
    for path in tqdm(chapter_links):
        response = requests.get(BASE_URL + path)
        html_content = response.text

        if 'no text in this page' in html_content:
            # print('No text.')
            continue

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove all tables
        tables = soup.findAll('table')
        for table in tables:
            # print('Removing table...')
            table.decompose()

        try:
            title = soup.find('span', class_='mw-page-title-main').text.strip()
        except:
            print('No title.')
            continue

        body = soup.find(id='mw-content-text')

        children = get_text_tags(body)

        headers = []
        sections = []

        for child in children:
            t = child.text.strip()
            if child.name == 'h2':
                if t.lower().startswith('reference') or t.lower().startswith('external links'):
                    break
                headers.append(t)
            elif child.name == 'p':
                sections.append(t)

        outputs= [f'# {title}']

        has_section = False
        for i in range(len(sections)):
            row = f'## {headers[i]}\n' if len(headers) == len(sections) else ''
            if sections[i] in seen:
                print(f'Section repeated: {sections[i]}')
            elif len(sections[i].split(' ')) >= MIN_SECTION_TOKENS:
                seen.add(sections[i])
                row += sections[i]
                outputs.append(row)
                has_section = True

        if has_section:
            clean_output = clean('\n\n'.join(outputs))
            num_tokens = len(re.split(r'\W+', clean_output))

            dataset.append({
                'id': str(title),
                'title': str(title),
                'text': clean_output,
                'num_tokens': num_tokens
            })
            print(f'Saving {num_tokens} tokens of content!')
        else:
            print('Found content but too short or could not parse.')

    hf_dir = os.path.join(OUT_DIR, 'dataset_hf')
    dataset = Dataset.from_list(dataset)
    
    print(f'Saving {len(dataset)} chapters to {hf_dir}')
    dataset.save_to_disk(hf_dir)
