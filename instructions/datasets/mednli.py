import json
import os

from datasets import Dataset, DatasetDict

MEDNLI_DIR = '/weka/home-griffin/clinical_instructions/mednli'
OUT_DIR = os.path.join(MEDNLI_DIR, 'dataset_hf')


INSTRUCTION = 'Assess whether or not a medical "hypothesis" logically follows from its "premise".'
QUESTION = 'Is the hypothesis entailed by the premise?'


LABEL_MAP = {
    'entailment': 'A',
    'contradiction': 'B',
    'neutral': 'C',
}

CHOICE_STR = 'A) yes\nB) no\nC) maybe'


if __name__ == '__main__':
    splits = ['train', 'validation', 'test']
    dataset = {}
    for split in splits:
        dataset[split] = []
        with open(os.path.join(MEDNLI_DIR, f'{split}.jsonl')) as fd:
            lines = fd.readlines()

            print(f'Loaded {len(lines)} examples from the {split.capitalize()} Set.')

            for line in lines:
                obj = json.loads(line)
                premise = obj['sentence1'].strip()
                hypothesis = obj['sentence2'].strip()
                completion = LABEL_MAP[obj['gold_label']]

                pieces = [f'# INSTRUCTION\n{INSTRUCTION}']
                pieces.append(f'# CONTEXT\nPremise: {premise}\nHypothesis: {hypothesis}')
                pieces.append(f'# QUESTION\n{QUESTION}')
                pieces.append(f'# CHOICES\n{CHOICE_STR}')
                pieces.append('# ANSWER\n')

                prompt = '\n\n'.join(pieces)

                dataset[split].append({
                    'id': obj['pairID'],
                    'prompt': prompt,
                    'completion': completion,
                    'num_options': len(LABEL_MAP),
                })
    
        dataset[split] = Dataset.from_list(dataset[split])
    dataset = DatasetDict(dataset)
    print(f'Saving MedNLI to {OUT_DIR}')
    dataset.save_to_disk(OUT_DIR)
