import os

from dataclasses import dataclass
from typing import Callable

import argparse
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from instructions.datasets.utils import *


OUT_DIR = '/weka/home-griffin/clinical_instructions/multimedqa'
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class MultiMedQAConfigs:
    name: str
    hf_args: tuple
    instruction: str
    input_to_prompt: Callable
    input_to_target: Callable
    num_options: int
    input_to_id: Callable = input_to_id_default
    train_split: str = 'train'
    validation_split: str = 'validation'
    test_split: str = 'test'
    cot_col: str = None


ALL_CONFIGS = [
    MultiMedQAConfigs(
        name='pubmedqa_artificial',
        hf_args=('bigbio/pubmed_qa', 'pubmed_qa_artificial_source'),
        instruction='Answer this Yes/No question using the following PubMed abstract as evidence by writing the letter associated with the correct answer.',
        input_to_prompt=input_to_prompt_pubmedqa,
        input_to_target=input_to_target_pubmedqa_artificial,
        cot_col='LONG_ANSWER',
        num_options=2
    ),
    MultiMedQAConfigs(
        name='pubmedqa_labeled',
        hf_args=('bigbio/pubmed_qa', 'pubmed_qa_labeled_fold0_source'),
        instruction='Answer this Yes/No/Maybe question using the following PubMed abstract as evidence by writing the letter associated with the correct answer.',
        input_to_prompt=input_to_prompt_pubmedqa,
        input_to_target=input_to_target_pubmedqa,
        cot_col='LONG_ANSWER',
        num_options=3
    ),
    MultiMedQAConfigs(
        name='medmcqa',
        hf_args=('/weka/home-griffin/clinical_instructions/medmcqa/dataset_hf', ),
        instruction='Answer this multiple-choice question on {} from the AIIMS & NEET PG entrance medical licensing exams by writing the letter associated with the correct answer.',
        input_to_prompt=input_to_prompt_medmcqa,
        input_to_target=input_to_target_medmcqa,
        cot_col='exp',
        num_options=4,
    ),
    MultiMedQAConfigs(
        name='medqa',
        hf_args=('GBaker/MedQA-USMLE-4-options-hf', ),
        instruction='Answer this multiple-choice question from the United States Medical Licensing Examination (USMLE) by writing the letter associated with the correct answer.',
        input_to_prompt=input_to_prompt_medqa,
        input_to_target=input_to_target_medqa,
        num_options=4
    ),
    MultiMedQAConfigs(
        name='mmlu_anatomy',
        hf_args=('hails/mmlu_no_train', 'anatomy'),
        instruction='Answer this multiple-choice question on Anatomy.',
        input_to_prompt=input_to_prompt_mmlu,
        input_to_target=input_to_target_mmlu,
        num_options=4,
        train_split='dev',
    ),
    MultiMedQAConfigs(
        name='mmlu_clinical_knowledge',
        hf_args=('hails/mmlu_no_train', 'clinical_knowledge'),
        instruction='Answer this multiple-choice question on Clinical Knowledge.',
        input_to_prompt=input_to_prompt_mmlu,
        input_to_target=input_to_target_mmlu,
        num_options=4,
        train_split='dev',
    ),
    MultiMedQAConfigs(
        name='mmlu_college_medicine',
        hf_args=('hails/mmlu_no_train', 'college_medicine'),
        instruction='Answer this multiple-choice question on College Medicine.',
        input_to_prompt=input_to_prompt_mmlu,
        input_to_target=input_to_target_mmlu,
        num_options=4,
        train_split='dev',
    ),
    MultiMedQAConfigs(
        name='mmlu_genetics',
        hf_args=('hails/mmlu_no_train', 'medical_genetics'),
        instruction='Answer this multiple-choice question on Genetics.',
        input_to_prompt=input_to_prompt_mmlu,
        input_to_target=input_to_target_mmlu,
        num_options=4,
        train_split='dev',
    ),
    MultiMedQAConfigs(
        name='mmlu_professional_medicine',
        hf_args=('hails/mmlu_no_train', 'professional_medicine'),
        instruction='Answer this multiple-choice question on Professional Medicine.',
        input_to_prompt=input_to_prompt_mmlu,
        input_to_target=input_to_target_mmlu,
        num_options=4,
        train_split='dev',
    ),
    MultiMedQAConfigs(
        name='mmlu_college_biology',
        hf_args=('hails/mmlu_no_train', 'college_biology'),
        instruction='Answer this multiple-choice question on College Biology.',
        input_to_prompt=input_to_prompt_mmlu,
        input_to_target=input_to_target_mmlu,
        num_options=4,
        train_split='dev',
    ),
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pre-processing MultiMedQA training dataset.')

    parser.add_argument('-add_cot', default=False, action='store_true')
    parser.add_argument('-add_artificial_pubmedqa', default=False, action='store_true')

    args = parser.parse_args()

    suffix = '_artificial' if args.add_artificial_pubmedqa else ''

    if args.add_cot:
        out_dir = os.path.join(OUT_DIR, f'dataset_cot_hf{suffix}')
    else:
        out_dir = os.path.join(OUT_DIR, f'dataset_hf{suffix}')

    if os.path.exists(out_dir):
        print(f'{out_dir} exists already. Before re-running, run "rm -rf {out_dir}"')
        exit(0)

    outputs = {
        'train': [],
        'validation': [],
        'test': []
    }

    for config in ALL_CONFIGS:
        if config.name == 'pubmedqa_artificial' and not args.add_artificial_pubmedqa:
            print(f'Skipping {config.name}...')
            continue
        print(f'Processing {config.name}...')
        try:
            dataset = load_from_disk(*config.hf_args)
        except:
            print('Loading from the Hub...')
            dataset = load_dataset(*config.hf_args)
        
        for split in ['train', 'validation', 'test']:
            config_split_name = getattr(config, f'{split}_split')
            if config_split_name is None:
                print(f'Make sure there really is no {config_split_name} for {config.name}...')
                continue
            elif config_split_name not in dataset:
                print(f'{config.name} has no {config_split_name} split. Make sure you have correct split names.')
                print(f'Available options are --> ' + ', '.join(list(dataset.keys())))
                continue

            data_split = dataset[config_split_name]
            for idx, example in enumerate(data_split):
                id = config.input_to_id(config.name, example, split, idx)

                if config.name == 'medmcqa':
                    instruction = config.instruction.format(example['subject_name'])
                else:
                    instruction = config.instruction

                explanation = example[config.cot_col] if config.cot_col is not None and args.add_cot else ''
                if explanation is not None and args.add_cot and len(explanation) > 0:  # For MedMCQA sometimes they are none
                    instruction += ' Explain your answer.'
                    is_cot = True
                else:
                    explanation = ''
                    is_cot = False
                prompt = f'# INSTRUCTION\n{instruction}\n\n{config.input_to_prompt(example, is_cot=is_cot)}'
                label, completion = config.input_to_target(example, explanation)

                out_row = {
                    'id': id,
                    'source': config.name,
                    'prompt': prompt,
                    'completion': completion,
                    'label': label,
                    'explanation': explanation,
                    'num_options': config.num_options,
                }

                outputs[split].append(out_row)
    
    for k, v in outputs.items():
        outputs[k] = Dataset.from_list(v)
    
    outputs = DatasetDict(outputs)

    print(f'Saving multimedqa training set to {out_dir}')
    outputs.save_to_disk(out_dir)
