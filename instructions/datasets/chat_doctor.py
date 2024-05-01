import os
import json

from datasets import Dataset
from tqdm import tqdm


CHAT_DOCTOR_DIR = '/weka/home-griffin/clinical_instructions/ChatDoctor'
os.makedirs(CHAT_DOCTOR_DIR, exist_ok=True)
IN_FN = os.path.join(CHAT_DOCTOR_DIR, 'HealthCareMagic-100k.json')
INSTRUCTION = 'A patient is asking you a health-related question. Answer it carefully and politely.'
OUT_DIR = os.path.join(CHAT_DOCTOR_DIR, 'dataset_hf')


def create_prompt(row):
    input = row['input']
    lines = [f'# INSTRUCTION\n{INSTRUCTION}', f'# QUESTION\n{input}', '# ANSWER\n']
    return '\n\n'.join(lines)


if __name__ == '__main__':
    dataset = []
    seen = set()

    if os.path.exists(OUT_DIR):
        print(OUT_DIR + 'already exists. Remove first.')
        exit(0)

    print(f'Opening dialogues from {IN_FN}')

    with open(IN_FN, 'r') as fd:
        dialogues = json.load(fd)

        for idx, row in tqdm(enumerate(dialogues), total=len(dialogues)):
            output_row = {
                'id': f'healthcare-magic-{idx}',
                'prompt': create_prompt(row),
                'completion': row['output']
            }

            assert output_row['prompt'] not in seen
            seen.add(output_row['prompt'])
            dataset.append(output_row)

    dataset = Dataset.from_list(dataset)
    print(f'Saving {len(dataset)} doctor-patient dialogue turns as QA pairs to {OUT_DIR}')
    dataset.save_to_disk(OUT_DIR)
