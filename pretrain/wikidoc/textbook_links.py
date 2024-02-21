import requests

from bs4 import BeautifulSoup

BASE_URL = 'https://www.wikidoc.org'
MAIN_URL = BASE_URL + '/index.php/Main_Page'
PREFIXES = [
    '/index.php/The_WikiDoc_Living_Textbook',
    '/index.php/Category',
    '/index.php/Basic_science_curriculum',
    '/index.php/COVID-19',
    '/index.php/Primary_care_status_update',
]


def get_links(href):
    response = requests.get(BASE_URL + href)
    html_content = response.text

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    element = soup.find(id='wikidocNav')
    if element:
        print('Removing navigation bar links...')
        element.decompose()

    # Find all <a> tags which typically denote links
    links = soup.find_all('a')
    # Should be an internal reference not href that starts with "https"
    return [l for l in links if l.get('href') is not None and l.get('href').startswith('/')]


def is_content_href(href):
    if 'template' in href.lower():
        return False

    if 'www' in href.lower():
        return False

    if 'special' in href.lower():
        return False

    if 'wikidoc' in href.lower():
        return False
    
    if '.org' in href.lower():
        return False
    
    if 'main_page' in href.lower():
        return False

    if '.svg' in href.lower():
        return False
    
    if '.png' in href.lower():
        return False

    if 'file:' in href.lower():
        return False

    if '.jpeg' in href.lower():
        return False

    if '.jpg' in href.lower():
        return False

    # These all say "no text" in them when you click
    if 'list of' in href.lower():
        return False
    
    # These all say "no text" in them when you click
    if 'list_of' in href.lower():
        return False

    if 'user:' in href.lower():
        return False
    
    # No query strings
    if '?' in href:
        return False

    print('Valid? -> ', href)
    return True


def extract_chapter_hrefs(href, seen):
    seen.add(href)
    # print('Processsing ', href)
    all_links = get_links(href)
    chapter_links = set()
    for link in all_links:
        sub_href = link.get('href')
        if sub_href in seen:
            continue
        matching_prefixes = [prefix for prefix in PREFIXES if sub_href.startswith(prefix)]
        if len(matching_prefixes) == 0:
            # Is this a content link?
            has_content = is_content_href(sub_href)
            if has_content:
                chapter_links.add(sub_href)
        elif sub_href not in seen:
            chapter_links = chapter_links.union(extract_chapter_hrefs(sub_href, seen=seen))
    return chapter_links


if __name__ == '__main__':
    # Fetch the HTML content of the webpage
    response = requests.get(MAIN_URL)
    html_content = response.text

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all <a> tags which typically denote links
    links = soup.find_all('a')

    chapter_hrefs = set()
    seen = set()
    # Extract the href attribute from each <a> tag
    for link in links:
        href = link.get('href')
        if href is None:
            continue

        if href is not None and any([href.startswith(prefix) for prefix in PREFIXES]):
            chapter_hrefs = chapter_hrefs.union(set(extract_chapter_hrefs(href, seen=seen)))

    chapter_hrefs = list(sorted(chapter_hrefs))
    print('\n'.join(chapter_hrefs))
    print(len(chapter_hrefs))

    with open('./chapter_links.txt', 'w') as fd:
        fd.write('\n'.join(chapter_hrefs))
