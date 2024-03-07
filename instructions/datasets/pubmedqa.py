
_URLS = {
    "pubmed_qa_artificial": "https://drive.google.com/uc?export=download&id=1kaU0ECRbVkrfjBAKtVsPCRF6qXSouoq9",
    "pubmed_qa_labeled": "https://drive.google.com/uc?export=download&id=1kQnjowPHOcxETvYko7DRG9wE7217BQrD",
    "pubmed_qa_unlabeled": "https://drive.google.com/uc?export=download&id=1q4T_nhhj8UvJ9JbZedhkTZHN6ZeEZ2H9",
}
import json
import os

from datasets import Dataset, DatasetDict


def load_artificial():
    dir = '/weka/home-griffin/clinical_instructions/multimedqa/pubmedqa'
    return DatasetDict({
        'train': generate_examples(os.path.join(dir, 'pqaa_train_set.json')),
        'valdation': generate_examples(os.path.join(dir, 'pqaa_dev_set.json')),
    })


def load_labeled():
    dir = '/weka/home-griffin/clinical_instructions/multimedqa/pubmedqa'
    fold_dir = os.path.join(dir, 'pqal_fold0')
    return DatasetDict({
        'train': generate_examples(os.path.join(fold_dir, 'train_set.json')),
        'valdation': generate_examples(os.path.join(fold_dir, 'dev_set.json')),
        'test': generate_examples(os.path.join(dir, 'pqal_test_set.json')),
    })


def generate_examples(filepath, is_artificial=False):
    data = json.load(open(filepath, "r"))

    out = []
    for id, row in data.items():
        if is_artificial:
            row["YEAR"] = None
            row["reasoning_required_pred"] = None
            row["reasoning_free_pred"] = None

        row['id'] = id
    
        out.append(row)

    out = Dataset.from_list(out)
    return out


if __name__ == '__main__':
    labeled_dir = '/weka/home-griffin/clinical_instructions/multimedqa/pubmedqa/labeled_hf'
    artificial_dir = '/weka/home-griffin/clinical_instructions/multimedqa/pubmedqa/artificial_hf'
    
    load_labeled().save_to_disk(labeled_dir)
    load_artificial().save_to_disk(artificial_dir)