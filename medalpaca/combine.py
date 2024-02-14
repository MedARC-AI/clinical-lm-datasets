from dataclasses import dataclass
from typing import Callable

from datasets import load_dataset


@dataclass
class MEDALPACA_CONFIG:
    name: str
    hf_path: str
    instruction: str
    split: str = 'train'


CONFIGS = [
    MEDALPACA_CONFIG(
        name='flashcards',
        hf_path='medalpaca/medical_meadow_medical_flashcards',
        instruction='As a medical student, generate a question-answer flashcard in the style of the Anki Medical Curriculum.'
    ),
    MEDALPACA_CONFIG(
        name='wikidoc_patient',
        hf_path='medalpaca/medical_meadow_wikidoc_patient_information',
        instruction='As a contributor to WikiDoc, ask a medical question and provide a paragraph-long answer.'
    ),
    # MEDALPACA_CONFIG(
    #     name='wikidoc_textbook',
    #     hf_path='medalpaca/medical_meadow_wikidoc',
    #     instruction='As a contributor to WikiDoc, ask a medical question and provide a paragraph-long answer.'
    # ),
    MEDALPACA_CONFIG(
        name='stack_exchange',
        hf_path='medalpaca/xx',
        instruction='As a contributor to Stack Exchange, ask a medical question and provide a top-rated answer.'
    )
]

if __name__ == '__main__':
    combined = []

    for dataset in DATASETS:
        pass
