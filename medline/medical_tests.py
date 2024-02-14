import os
import regex as re
import string
from tqdm import tqdm
import requests

from bs4 import BeautifulSoup
from datasets import Dataset 


MIN_SUMMARY_TOKENS = 25
ALL_TESTS_URL = 'https://medlineplus.gov/lab-tests/'
OUT_DIR = os.path.join('/weka/home-griffin/clinical_pile/medline')
os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == '__main__':
    response = requests.get(ALL_TESTS_URL)
    html_content = response.text

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser').find('div', class_='main')

    letters = list(string.ascii_uppercase) + ['0-9']

    links = set()
    link2name = {}
    for letter in letters:
        div = soup.find('div', attrs={'id': f'section_{letter}'})
        if div is None:
            assert letter in {'J', 'Q'}
            continue
        # Find all <a> tags which typically denote links
        letter_links = div.find_all('a')
        for link in letter_links:
            links.add(link.get('href'))
            link2name[link.get('href')] = link.text.strip()

    dataset = []
    seen = set()
    for link in tqdm(sorted(links)):
        topic = link2name[link]
        response = requests.get(link)
        html_content = response.text

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        main = soup.find('div', class_='main')

        sections = main.find_all('section')

        if 'references' in sections[-1].text.lower():
            sections = sections[:-1]

        lines = [f'# {topic}\n']

        for section in sections:
            for child in section.find('div', class_='mp-content').children:
                if child.name is None:
                    if len(child.text.strip()) > 0:
                        print('No tag name...')
                        lines.append(child.text.strip())
                elif child.name.startswith('h'):
                    lines.append('\n## ' + child.text.strip())
                elif child.name == 'p':
                    lines.append(child.text.strip())
                elif child.name in {'ul', 'ol'}:
                    for item in child.find_all('li'):
                        lines.append(f'- {item.text.strip()}')
            lines.append('\n')

        text = '\n'.join(lines)

        text = re.sub('\n\n\n', '\n\n', text).strip()
        num_tokens = len(re.split(r'\W+', text))

        if num_tokens < MIN_SUMMARY_TOKENS:
            print(f'Too short -->\n{text}')
        else:
            row = {
                'id': f'medline-test-{topic}',
                'topic': topic,
                'num_tokens': num_tokens,
                'text': text,
            }
            dataset.append(row)
    
    dataset = Dataset.from_list(dataset)
    out_dir = os.path.join(OUT_DIR, 'medical_tests_hf')
    print(f'Saving {len(dataset)} medical test articles to {out_dir}')
    dataset.save_to_disk(out_dir)
