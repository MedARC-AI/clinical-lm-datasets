
import os
import regex as re
import string
from tqdm import tqdm
import requests

from bs4 import BeautifulSoup
from datasets import Dataset 


MIN_ARTICLE_TOKENS = 50
OUT_DIR = os.path.join('/weka/home-griffin/clinical_pile/medline')
os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == '__main__':
    # This is "A" conditions
    links = ['https://medlineplus.gov/genetics/condition/']
    for l in list(string.ascii_lowercase)[1:] + ['0']:
        links.append(f'https://medlineplus.gov/genetics/condition-{l}')
    
    condition_links = set()
    link2condition = {}
    for link in links:
        response = requests.get(link)
        html_content = response.text

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser').find('div', class_='main-one')

        element = soup.find('nav')
        if element:
            print('Removing navigation bar links...')
            element.decompose()
        
        ct = 0
        for a in soup.find_all('a'):
            ct += 1
            condition_links.add(a.get('href'))
            link2condition[a.get('href')] = a.text.strip()
        
        print(f'{link} -> {ct}')

    dataset = []
    seen = set()
    for link in tqdm(sorted(list(condition_links))):
        condition = link2condition[link]
        response = requests.get(link)
        html_content = response.text

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        main = soup.find('div', class_='main')

        chapters = main.find_all('div', class_='mp-exp')

        if 'references' in chapters[-1].text.lower():
            chapters = chapters[:-1]

        lines = [f'# {condition}\n']

        for idx, chapter in enumerate(chapters):
            header_div = chapter.find('div', class_='exp-header')
            if header_div is None:
                header_div = chapter.find('h2')
            if header_div is None:
                header_div = chapter.find('h3')

            header = header_div.text.strip()
            header_div.decompose()

            if 'additional information' in header.lower():
                continue

            if 'references' in header.lower():
                continue

            if 'learn more' in header.lower():
                continue
            
            chapter_out = [f'\n## {header}']

            body = chapter.find('div', class_='exp-body')
            if body is None:
                body = chapter
            sections = body.find_all('section')

            if len(sections) == 0:
                continue

            for section in sections:
                if section.find('div', class_='mp-content') is not None:
                    section = section.find('div', class_='mp-content')

                for para in section.children:
                    if para.name is None:
                        if len(para.text.strip()) > 0:
                            print('No tag name...')
                            chapter_out.append(para.text.strip())
                    elif para.name == 'p':
                        chapter_out.append(para.text.strip())
                    elif para.name in {'ul', 'ol'}:
                        for item in para.find_all('li'):
                            chapter_out.append(f'- {item.text.strip()}')
                    # else:
                    #     print(f'Skipping -> {para.text.strip()}')
                chapter_out.append('\n')
            
            num_chapter_toks = len(' '.join(chapter_out).split(' '))
            if num_chapter_toks >= 5:
                lines += chapter_out
            # else:
            #     print('Chapter (below) is too short...')
            #     print('\n'.join(chapter_out))

        text = '\n'.join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        num_tokens = len(re.split(r'\W+', text))

        print(text)

        if num_tokens < MIN_ARTICLE_TOKENS:
            print(f'Too short -->\n{text}')
        else:
            row = {
                'id': f'medline-genetic-condition-{condition}',
                'condition': condition,
                'num_tokens': num_tokens,
                'text': text,
            }
            dataset.append(row)

    dataset = Dataset.from_list(dataset)
    out_dir = os.path.join(OUT_DIR, 'genetic_conditions_hf')
    print(f'Saving {len(dataset)} genetic condition articles to {out_dir}')
    dataset.save_to_disk(out_dir)
