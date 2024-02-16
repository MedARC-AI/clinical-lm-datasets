import os
import json

from datasets import Dataset


CHAT_DOCTOR_DIR = '/weka/home-griffin/clinical_pile/ChatDoctor/'
IN_FN = os.path.join(CHAT_DOCTOR_DIR, 'HealthCareMagic-100k.json')
INSTRUCTION = 'A patient is asking you a health-related question. Answer it carefully and politely.'
OUT_DIR = os.path.join(CHAT_DOCTOR_DIR, 'dataset_hf')


def create_prompt(row):
    input = row['input']
    return f'<<Instruction:>> {INSTRUCTION}\n----\n<<Question:>> {input}\n----\n<<Answer:>> '


if __name__ == '__main__':
    dataset = []
    with open(IN_FN, 'r') as fd:
        dialogues = json.load(fd)

        for row in dialogues:
            output_row = {
                'prompt': create_prompt(row),
                'completion': row['output']
            }

            dataset.append(output_row)
    
    dataset = Dataset.from_list(dataset)
    print(f'Saving {len(dataset)} doctor-patient dialogue turns as QA pairs to {OUT_DIR}')
    dataset.save_to_disk(OUT_DIR)
