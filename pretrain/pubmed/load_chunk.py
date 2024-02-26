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
    pubmed_path = os.path.join(dataset_dir, f"{dataset}-PubMed_v2.jsonl")
    # other_path = os.path.join(dataset_dir, f"{dataset}-nonPubMed.jsonl")
    if os.path.exists(pubmed_path):  # or os.path.exists(other_path):
        print(f'Removing existing {pubmed_path}')
        os.remove(pubmed_path)
    print(f'\n4. Filtering {dataset} dataset at {dataset_path} into PubMed and non-PubMed articles.\n')
    pubmed_count = 0
    other_count = 0
    with open(dataset_path, 'r') as f_in:
        for line in tqdm(f_in):
            # if '"PubMedCentral":null' in line:
            #     other_count += 1
            #     # article = json.loads(line)
            #     # pmc_id = get_ids(article)

            #     # assert pmc_id is None
            # else:
            article = json.loads(line)
            pmc_id = get_ids(article)
            if pmc_id is not None:
                # f_pubmed.write(json.dumps(article) + "\n")
                pubmed_count += 1
                if pubmed_count % 10000 == 0:
                    print(f"So Far: {pubmed_count} PubMed articles and {other_count} non-PubMed articles.\n")
            else:
                other_count += 1

    # with open(dataset_path, 'r') as f_in, open(pubmed_path, 'w') as f_pubmed:
    #     for line in tqdm(f_in):
    #         # if '"PubMedCentral":null' in line:
    #         #     other_count += 1
    #         #     # article = json.loads(line)
    #         #     # pmc_id = get_ids(article)

    #         #     # assert pmc_id is None
    #         # else:
    #         article = json.loads(line)
    #         pmc_id = get_ids(article)
    #         if pmc_id is not None:
    #             f_pubmed.write(json.dumps(article) + "\n")
    #             pubmed_count += 1
    #             if pubmed_count % 10000 == 0:
    #                 print(f"So Far: {pubmed_count} PubMed articles and {other_count} non-PubMed articles.\n")
    #         else:
    #             other_count += 1
            #     f_other.write(json.dumps(article) + "\n")
    print(f"Finished filtering {dataset} dataset into PubMed and non-PubMed articles.")
    print(f"Found {pubmed_count} PubMed articles and {other_count} non-PubMed articles.\n")


if __name__ == '__main__':
    data_path = '/weka/home-griffin/clinical_pile/pubmed/abstracts/'
    dataset = 'abstracts'
    filter_pubmed(data_path, dataset)
