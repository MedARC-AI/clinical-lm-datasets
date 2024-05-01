import os

from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class MEDALPACA_CONFIG:
    name: str
    hf_dir: str
    instruction: str
    out_dir: str
    split: str = 'train'


def create_prompt(row, config):
    input = row['input']
    lines = [f'# INSTRUCTION\n{config.instruction}', f'# QUESTION\n{input}', '# ANSWER\n']
    return '\n\n'.join(lines)


CONFIGS = [
    MEDALPACA_CONFIG(
        name='alpaca_flashcards',
        hf_dir='medalpaca/medical_meadow_medical_flashcards',
        instruction='As a medical student, answer a flashcard question in the style of the Anki Medical Curriculum.',
        out_dir='/weka/home-griffin/clinical_instructions/medalpaca/flashcards_hf'
    ),
    MEDALPACA_CONFIG(
        name='wikidoc_patient',
        hf_dir='medalpaca/medical_meadow_wikidoc_patient_information',
        instruction='As a contributor to WikiDoc, provide a paragraph-long answer to a patient\'s medical question.',
        out_dir='/weka/home-griffin/clinical_instructions/medalpaca/wikidoc_patient_hf'
    ),
]


def construct_example(row, idx, config):
    prompt = create_prompt(row, config)
    completion = row['output']
    assert type(completion) == str
    if len(completion) > 0:
        return {
            'id': config.name + '-' + str(idx),
            'source': config.name,
            'prompt': prompt,
            'completion': completion,
        }
    return {'id': None, 'source': None, 'prompt': None, 'completion': None}


if __name__ == '__main__':
    for config in CONFIGS:
        if os.path.exists(config.out_dir):
            print(config.out_dir + ' already exists. Remove first.')
            exit(0)

        print(config.hf_dir)
        data = load_dataset(config.hf_dir)

        transformed = data.map(
            lambda row, idx: construct_example(row, idx, config),
            with_indices=True,
            remove_columns=['instruction', 'input', 'output']
        )

        prev_n = len(transformed['train'])

        transformed = transformed.filter(lambda row: row['id'] is not None)

        new_n = len(transformed['train'])

        print(f'{prev_n - new_n}/{prev_n} examples had no completions')

        print(f'Saving {new_n} examples from {config.hf_dir} to {config.out_dir}')
        transformed.save_to_disk(config.out_dir)
