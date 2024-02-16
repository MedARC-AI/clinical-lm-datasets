import json
import os

from datasets import Dataset, DatasetDict

MEDNLI_DIR = '/weka/home-griffin/clinical_pile/mednli'
OUT_DIR = '/weka/home-griffin/clinical_pile/mednli/dataset_hf'


INSTRUCTION = 'Does the given medical "hypothesis" logically follow from the "premise"?'


LABEL_MAP = {
    'entailment': 'yes',
    'contradiction': 'no',
    'neutral': 'maybe',
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
                print(obj)

                premise = obj['sentence1']
                hypothesis = obj['sentence2']
                

                prompt = f'<<Instruction:>> {INSTRUCTION}\n----\n<<Premise:>> {premise}\n----\n<<Hypothesis:>> {hypothesis}\n----\n<<Choices:>>\n{CHOICE_STR}\n----\n<<Answer:>> '
                completion = LABEL_MAP[obj['gold_label']]

                dataset[split].append({
                    'id': obj['pairID'],
                    'prompt': prompt,
                    'completion': completion
                })
    
        dataset[split] = Dataset.from_list(dataset[split])
    dataset = DatasetDict(dataset)
    print(f'Saving MedNLI to {OUT_DIR}')
    dataset.save_to_disk(OUT_DIR)
