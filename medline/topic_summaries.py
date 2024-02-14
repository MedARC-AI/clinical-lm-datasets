import os
import regex as re
import string
from tqdm import tqdm
import requests

from bs4 import BeautifulSoup
from datasets import Dataset 


MIN_SUMMARY_TOKENS = 25
ALL_TOPIC_URL = 'https://medlineplus.gov/all_healthtopics.html'
OUT_DIR = os.path.join('/weka/home-griffin/clinical_pile/medline')
os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == '__main__':
    response = requests.get(ALL_TOPIC_URL)
    html_content = response.text

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser').find('article')

    letters = string.ascii_uppercase

    links = set()
    link2name = {}
    for letter in letters:
        div = soup.find('div', attrs={'id': f'section_{letter}'})
        # Find all <a> tags which typically denote links
        letter_links = div.find_all('a')
        for link in letter_links:
            links.add(link.get('href'))
            link2name[link.get('href')] = link.text.strip()

    summaries = []
    seen = set()
    for link in tqdm(sorted(links)):
        topic = link2name[link]
        response = requests.get(link)
        html_content = response.text

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        summary = soup.find('div', {'id': 'topic-summary'})

        # remove class 'attribution'
        attribution = soup.find(class_='attribution')
        if attribution:
            # print('Removing attribution of sources.')
            attribution.decompose()

        lines = [f'# Summary of {topic}\n']

        for child in summary.children:    
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

        text = '\n'.join(lines)
        text = re.sub('\n\n\n', '\n\n', text)
        num_tokens = len(re.split(r'\W+', text))

        if num_tokens < MIN_SUMMARY_TOKENS:
            print(f'Too short -->\n{text}')
        else:
            row = {
                'id': f'medline-topic-{topic}',
                'topic': topic,
                'num_tokens': num_tokens,
                'text': text,
            }
            # print(row['text'])
            summaries.append(row)
    
    summaries = Dataset.from_list(summaries)
    out_dir = os.path.join(OUT_DIR, 'topic_summaries_hf')
    print(f'Saving {len(summaries)} summaries to {out_dir}')
    summaries.save_to_disk(out_dir)
