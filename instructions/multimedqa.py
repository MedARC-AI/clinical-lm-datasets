import os

from dataclasses import dataclass
from typing import Callable

from datasets import Dataset, DatasetDict, load_dataset
from utils import *


OUT_DIR = '/weka/home-griffin/clinical_instructions/multimedqa'
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class MultiMedQAConfigs:
    name: str
    hf_args: tuple
    instruction: str
    input_to_prompt: Callable
    input_to_target: Callable
    input_to_id: Callable = input_to_id_default
    train_split: str = 'train'
    validation_split: str = 'validation'
    test_split: str = 'test'


ALL_CONFIGS = [
    MultiMedQAConfigs(
        name='pubmedqa',
        hf_args=('bigbio/pubmed_qa', 'pubmed_qa_labeled_fold0_source'),
        instruction='Answer this multiple-choice question using the following PubMed abstract as evidence by writing the letter associated with the correct answer.',
        input_to_prompt=input_to_prompt_pubmedqa,
        input_to_target=input_to_target_pubmedqa,
    ),
    MultiMedQAConfigs(
        name='medmcqa',
        hf_args=('medmcqa', ),
        instruction='Answer this multiple-choice question from the AIIMS & NEET PG entrance medical licensing exams by writing the letter associated with the correct answer.',
        input_to_prompt=input_to_prompt_medmcqa,
        input_to_target=input_to_target_medmcqa,
    ),
    MultiMedQAConfigs(
        name='medqa',
        hf_args=('GBaker/MedQA-USMLE-4-options-hf', ),
        instruction='Answer this multiple-choice question from the United States Medical Licensing Examination (USMLE) by writing the letter associated with the correct answer.',
        input_to_prompt=input_to_prompt_medqa,
        input_to_target=input_to_target_medqa,
    ),
]


if __name__ == '__main__':
    outputs = {
        'train': [],
        'validation': [],
        'test': []
    }

    for config in ALL_CONFIGS:
        dataset = load_dataset(*config.hf_args)

        for split in ['train', 'validation', 'test']:
            data_split = dataset[getattr(config, f'{split}_split')]
            for idx, example in enumerate(data_split):
                id = config.input_to_id(example, split, idx)
                prompt = f'<<Instruction:>> {config.instruction}\n----\n{config.input_to_prompt(example)}'
                # Important Make sure prompt has a trailing space
                prompt = prompt.strip() + ' '
                completion = config.input_to_target(example)

                out_row = {
                    'id': id,
                    'task': config.name,
                    'prompt': prompt,
                    'completion': completion
                }

                outputs[split].append(out_row)
    
    for k, v in outputs.items():
        outputs[k] = Dataset.from_list(v)
    
    outputs = DatasetDict(outputs)

    out_dir = os.path.join(OUT_DIR, 'dataset_hf')
    print(f'Saving multimedqa training set to {out_dir}')
    # outputs.save_to_disk(out_dir)

    outputs.push_to_hub('medarc/sft_multimedqa')
