from dataclasses import dataclass
from typing import Callable

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
    return f'<<Instruction:>> {config.instruction}\n----\n<<Question:>> {input}\n----\n<<Answer:>> '


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
    # MEDALPACA_CONFIG(
    #     name='wikidoc_textbook',
    #     hf_path='medalpaca/medical_meadow_wikidoc',
    #     instruction='As a contributor to WikiDoc, ask a medical question and provide a paragraph-long answer.'
    # ),
    # MEDALPACA_CONFIG(
    #     name='stack_exchange',
    #     hf_path='medalpaca/xx',
    #     instruction='As a contributor to Stack Exchange, ask a medical question and provide a top-rated answer.'
    # )
]


def construct_example(row, idx, config):
    prompt = create_prompt(row, config)
    completion = row['output']
    return {
        'id': config.name + '-' + str(idx),
        'source': config.name,
        'prompt': prompt,
        'completion': completion,
    }


if __name__ == '__main__':
    for config in CONFIGS:
        print(config.hf_dir)
        data = load_dataset(config.hf_dir)

        transformed = data.map(
            lambda row, idx: construct_example(row, idx, config),
            with_indices=True,
            remove_columns=['instruction', 'input', 'output']
        )

        print(f'Saving {len(transformed)} examples from {config.hf_dir} to {config.out_dir}')
        transformed.save_to_disk(config.out_dir)
