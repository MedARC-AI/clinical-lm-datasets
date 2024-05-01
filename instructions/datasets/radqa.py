import os
import json
import argparse
from datasets import Dataset, DatasetDict


INSTRUCTION = 'Analyze a Radiology Report and respond to a question with the most relevant snippet from the report. If the question is clinically impossible, write "Impossible".'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script Loads RadQA into HF format.')

    parser.add_argument('--data_dir', default='/weka/home-griffin/clinical_instructions/radqa')

    parser.add_argument('-debug', action='store_true', default=False,
                        help='Enable debug mode (defaults to False).')

    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, 'dataset_hf')

    if os.path.exists(out_dir):
        print(out_dir + ' already exists. Remove first.')
        exit(0)

    # Call the main function with the parsed arguments
    SPLITS = {
        'train': 'train',
        'validation': 'dev',
        'test': 'test'
    }

    out_data = {
        'train': [],
        'validation': [],
        'test': []
    }

    num_0 = 0
    num_1 = 0
    num_2 = 0

    for save_split, data_split in SPLITS.items():
        fn = os.path.join(args.data_dir, f'{data_split}.json')
        with open(fn) as fd:
            split = json.load(fd)['data']

            for row_idx, row in enumerate(split):
                paras = row['paragraphs']
                title = row['title']
                for para_idx, para in enumerate(paras):
                    context = para['context']
                    for qidx, qa in enumerate(para['qas']):
                        question = qa['question']
                        id = qa['id']
                        answers = list(set([x['text'] for x in qa['answers']]))

                        is_impossible = qa['is_impossible']
                        if is_impossible:
                            assert len(answers) == 0
                            num_0 += 1
                            answer = 'This is impossible.'
                        elif len(answers) == 1:
                            num_1 += 1
                            answer = answers[0]
                        else:
                            # For now just take 1st (most are 1)
                            answer = answers[0]
                            num_2 += 1

                        out_lines = [
                            f'# INSTRUCTION\n{INSTRUCTION}',
                            f'# CONTEXT\n{context}',
                            f'# QUESTION\n{question}',
                            '# ANSWER\n',
                        ]

                        prompt = '\n\n'.join(out_lines)

                        completion = answer

                        out_data[save_split].append({
                            'id': id,
                            'idxs': f'{row_idx}_{para_idx}_{qidx}',
                            'prompt': prompt,
                            'completion': completion,
                        })
    
    for split, data in out_data.items():
        out_data[split] = Dataset.from_list(data)
    
    print(num_0, num_1, num_2)

    out_data = DatasetDict(out_data)
    print(f'Saving dataset to {out_dir}...')
    out_data.save_to_disk(out_dir)
