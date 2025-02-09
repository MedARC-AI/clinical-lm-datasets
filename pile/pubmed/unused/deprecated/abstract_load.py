"""
This script implements the data preparation pipeline for the s2orc, papers and abstracts 
datasets from the Semantic Scholar API (https://www.semanticscholar.org/product/api)
Including: download, extraction, aggregation, filtering and metadata joining 

The resulting files are:
- path/s2orc-PubMed_metadata.jsonl: 
    PubMed or PubMedCentral full-text articles (open-access subset)
- path/abstracts-PubMed_metadata.jsonl:
    PubMed or PubMedCentral abstracts (open-access subset)
"""

import requests
import json
import os
import gzip
import shutil
import argparse
from tqdm import tqdm
from p_tqdm import p_uimap


def get_s2orc_credentials(path='keys.json'):
    """
    Load S2ORC API key from json file.
    """
    with open(path) as f:
        keys = json.load(f)
    global API_KEY
    API_KEY = keys['api_key']
    return API_KEY

def get_links(dataset): 
    '''
    Get all file links from the API.
    '''
    headers = {'Accept': '*/*', 'x-api-key': API_KEY}
    dataset_url = f"https://api.semanticscholar.org/datasets/v1/release/latest/dataset/{dataset}"
    response = requests.get(dataset_url, headers=headers)
    links = response.json()["files"]
    return links

def download_file(link, path): 
    '''
    Download a single file from the API link to the given path.
    '''
    with open(path, "wb") as f:
        response = requests.get(link, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        for chunk in response.iter_content(chunk_size=2048):
            if chunk:
                f.write(chunk)
                f.flush()
                progress_bar.update(len(chunk))
    progress_bar.close()

def download_dataset(data_path, dataset):
    '''
    Download all files in dataset from the Semantic Scholar API.
    '''
    try:
        # Create dataset directory if it doesn't exist
        print(f'\n1. Downloading {dataset} dataset from the Semantic Scholar API.\n')
        dataset_dir = os.path.join(data_path, dataset)
        if os.path.exists(dataset_dir):
            print(f"Found existing {dataset} directory at {dataset_dir}.")
        else:
            print(f"Creating {dataset} directory at {dataset_dir}.")
            os.mkdir(dataset_dir)

        # Get all dataset file links from the API
        links = get_links(dataset)
        num_files = len(links)
        print(f"Found {num_files} files to download in {dataset} dataset.")

        # Download all files from the API
        for i, file_link in enumerate(links):
            json_path = os.path.join(dataset_dir, f'{dataset}_{i+1}.jsonl')
            gz_path = os.path.join(dataset_dir, f'{dataset}_{i+1}.gz')
            if os.path.exists(gz_path) or os.path.exists(json_path):
                print(f"[{i+1} | {num_files}] File already downloaded, skipping.")
                continue
            print(f'[{i+1} | {num_files}] Downloading file {gz_path}.')
            try:
                download_file(file_link, gz_path)
            except KeyboardInterrupt:
                print("\nDownload interrupted by user. Deleting partially downloaded file.")
                if os.path.exists(gz_path):
                    os.remove(gz_path)
                raise  # Re-raise KeyboardInterrupt to exit the script
        print(f'Finished downloading {dataset} dataset.\n')
    except Exception as e:
        print(f"An error occurred during the download: {e}")

def extract_dataset(dataset_dir):
    '''
    Extract all .gz files in dataset_dir into .jsonl format.
    '''
    # Check if any .gz files exist in dataset_dir
    gz_paths = [os.path.join(dataset_dir, file_name) for file_name in os.listdir(dataset_dir) if file_name.endswith(".gz")]
    if len(gz_paths) == 0:
        return
    print(f'\n2. Extracting all .gz files in {dataset_dir} to .jsonl format.\n')
    for i, gz_path in enumerate(gz_paths):
        file_path = gz_path[:-3] + ".jsonl"
        print(f'[{i+1} | {len(gz_paths)}] Extracting file {gz_path} to .jsonl') 
        with gzip.open(gz_path, 'rb') as f_in:
            with open(file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
    print(f'Finished extracting {dataset_dir} files to .jsonl format.\n')

def combine_dataset(dataset_dir, dataset):
    '''
    Combine all files in paths into a single file.
    '''
    dataset_path = os.path.join(dataset_dir, f'{dataset}.jsonl')
    paths = [os.path.join(dataset_dir, file_name) for file_name in os.listdir(dataset_dir) if file_name.endswith(".jsonl")]
    if os.path.exists(dataset_path):
        return
    print(f'\n3. Combining {len(paths)} .jsonl files in {dataset_dir} into a single file: {dataset_path}.\n')
    print(f"Creating {dataset_path}.")
    with open(dataset_path, 'a') as f_out:
        for i, path in enumerate(paths):
            print(f'[{i+1} | {len(paths)}] Aggregating file {path}')
            with open(path, 'r') as f_in:
                for line in tqdm(f_in):
                    f_out.write(line)
    print(f'Finished aggregating {dataset_dir} files into {dataset_path}.\n')

def get_ids(article):
    '''
    Helper to extract PubMed and PMC IDs from a given article depending on dataset.
    '''
    try:
        ids = article.get('openaccessinfo').get('externalids')
        if ids:
            return ids.get('PubMedCentral')
    except:
        pass
    return None


def filter_pubmed(dataset_dir, dataset): 
    """
    Separate dataset into PubMed+PMC vs. non-PubMed articles.
    """
    dataset_path = os.path.join(dataset_dir, f"{dataset}.jsonl")
    if not os.path.exists(dataset_path):
        raise ValueError(f'Could not find {dataset} dataset at {dataset_path}.')
    pubmed_path = os.path.join(dataset_dir, f"{dataset}-PubMed.jsonl")
    if os.path.exists(pubmed_path):
        print(f'Removing existing {pubmed_path}')
        os.remove(pubmed_path)
    print(f'\n4. Filtering {dataset} dataset at {dataset_path} into PubMed and non-PubMed articles.\n')
    pubmed_count = 0
    other_count = 0
    with open(dataset_path, 'r') as f_in:
        for line in tqdm(f_in):
            article = json.loads(line)
            pmc_id = get_ids(article)
            if pmc_id is not None:
                pubmed_count += 1
                if pubmed_count % 10000 == 0:
                    print(f"So Far: {pubmed_count} PubMed articles and {other_count} non-PubMed articles.\n")
            else:
                other_count += 1

    print(f"Finished filtering {dataset} dataset into PubMed and non-PubMed articles.")
    print(f"Found {pubmed_count} PubMed articles and {other_count} non-PubMed articles.\n")


def join_metadata(papers_pm_path, dataset_pm_path): 
    """ Join PubMed articles (from s2orc or abstracts) with metadata from papers dataset. """
    print(f'\n5. Joining {dataset_pm_path} with metadata from {papers_pm_path}.\n')
    if not os.path.exists(papers_pm_path):
        raise ValueError(f'Could not find papers dataset at {papers_pm_path}. \
                            Please download the papers dataset before joining with metadata.')
    if not os.path.exists(dataset_pm_path):
        raise ValueError(f'Could not find {dataset_pm_path}. \
                            Please download and filter the dataset before joining with metadata.')
    out_path = dataset_pm_path.split('.')[0] + '_metadata.jsonl'
    total = 0
    print(f'Loading papers metadata from {papers_pm_path}.')

    papers = [json.loads(line) for line in tqdm(open(papers_pm_path, 'r'))]
    papers = {paper.get('corpusid'): paper for paper in papers}

    print(f'Joining {dataset_pm_path} with metadata from {papers_pm_path}.')
    with open(dataset_pm_path, 'r') as f_in, open(out_path, 'w') as f_out: 
        count = 0
        # records = list(p_uimap(json.loads, f_in))
        for line in tqdm(f_in):
            total += 1
            record = json.loads(line)
            corpus_id = record.get('corpusid')
            match = papers.get(corpus_id)
            if match:
                record.update(match)
                f_out.write(json.dumps(record) + '\n')
                count += 1

    print(f'Finished joining {dataset_pm_path} with metadata from {papers_pm_path}.\n')
    print(f'Joined {count} articles out of {total}.\n')


def data_pipeline(data_path, dataset):
    """ Run the full data pipeline for a given dataset. """
    print(f'\nData pipeline entered for {dataset} dataset.\n')

    # 1. Load dataset from API into dataset_dir
    dataset_dir = os.path.join(data_path, dataset)
    download_dataset(data_path, dataset)

    # 2. Extract all .gz files to .jsonl format (if necessary)
    extract_dataset(dataset_dir)

    # 3. Aggregate all .jsonl files for dataset into a single file
    combine_dataset(dataset_dir, dataset)

    # 4. Keep only PubMed or PMC articles for the given dataset
    filter_pubmed(dataset_dir, dataset)

    # 5. Join PubMed articles with metadata from papers dataset
    papers_pm_path = os.path.join(data_path, 'papers', 'papers-PubMed.jsonl')
    dataset_pm_path = os.path.join(dataset_dir, f'{dataset}-PubMed.jsonl')
    
    join_metadata(papers_pm_path, dataset_pm_path)

    # 6. Adapt abtracts keys
    dataset_meta_path = os.path.join(dataset_dir, f'{dataset}-PubMed_metadata.jsonl')
    tmp_path = dataset_meta_path + '.tmp'
    with open(dataset_meta_path, 'r') as f_in, open(tmp_path, 'w') as f_out:
        records = list(p_uimap(json.loads, f_in))
        for record in tqdm(records):
            # record = json.loads(line)
            record.pop('openaccessinfo')
            record['text'] = record.pop('abstract')
            f_out.write(json.dumps(record) + '\n')
    os.remove(dataset_meta_path)
    os.rename(tmp_path, dataset_meta_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default='/weka/home-griffin/clinical_pile/pubmed',
        help="Path to directory where we store data. Default to ./data.")
    parser.add_argument(
        "--key_path",
        type=str,
        default='/weka/home-griffin/clinical_pile/credentials/s2_api_key.json',
        help="Path to json file containing Semantic Scholar API key.")

    args = parser.parse_args()

    get_s2orc_credentials(args.key_path) 

    data_pipeline(args.data_path, 'abstracts')
