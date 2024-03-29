import os

import numpy as np
from datasets import load_from_disk
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


DATA_DIR = '/weka/home-griffin/clinical_pile/v1/dataset_hf_50k_sample_llm_quality_scores/'

if __name__ == '__main__':
    dataset = load_from_disk(os.path.join(DATA_DIR, 'dataset_hf'))

    labels = dataset['label']
    print(np.mean(labels))

    # Create histogram
    sns.histplot(labels, bins=5, kde=False)
    plt.xlabel('Likert Score')
    plt.ylabel('Mixtral Predictions')
    plt.title('Histogram of Likert Scores')

    hist_fn = os.path.join(DATA_DIR, 'likert_histogram.png')
    print(f'Saving Likert Histogram to {hist_fn}')
    plt.savefig(hist_fn)

    plt.close()

    source_data = []
    for source in list(set(dataset['source'])):
        sub = dataset.filter(lambda row: row['source'] == source)
        row = {'source': source, 'n': len(sub), 'avg_likert': float(np.mean(sub['label']))}
        source_data.append(row)

    source_data = list(sorted(source_data, key=lambda x: -x['avg_likert']))
    print(tabulate(source_data))
