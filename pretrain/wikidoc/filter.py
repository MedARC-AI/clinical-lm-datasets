import os
import requests
import regex as re
from pathlib import Path

import ftfy
from bs4 import BeautifulSoup
from datasets import Dataset
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


if __name__ == '__main__':
    ## Many pages with "There is currently no text in this page."
    script_dir = Path(__file__).resolve().parent
    filename = 'chapter_links.txt'
    file_path = script_dir / filename

    with open(file_path, 'r') as file:
        chapter_links = [x.strip() for x in file.readlines() if len(x) > 0]
    
    dataset = []
    seen = set()
    seen_titles = set()
    for path in tqdm(chapter_links):
        response = requests.get(BASE_URL + path)
        # print('\n\n')
        # print(BASE_URL + path)

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
            title = ''

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

        outputs = []
        if len(title) > 0:
            outputs.append(f'# {title}')

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

            if len(title) > 0:
                assert title not in seen_titles
                seen_titles.add(title)

            dataset.append({
                'id': path,
                'text': clean_output,
                'num_tokens': num_tokens
            })
            print(f'Saving {num_tokens} tokens of content!')
            # print('\n\n')
            # print(clean_output)
            # print('\n\n')
        else:
            print('Found content but too short or could not parse.')

    hf_dir = os.path.join(OUT_DIR, 'dataset_hf')
    dataset = Dataset.from_list(dataset)
    # TODO put back
    # print(f'Saving {len(dataset)} chapters to {hf_dir}')
    # dataset.save_to_disk(hf_dir)
