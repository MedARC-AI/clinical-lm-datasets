from datasets import concatenate_datasets, load_dataset

import os
import argparse
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm


SUBSETS = [
    'wikihow',
    'auto_math_text',
    'khanacademy',
    'openstax',
    'stanford',
    # 'stories',
    # 'web_samples_v1',
    'web_samples_v2',
]


QUALITY_LABELS = {
    'ok', 'good', 'very good',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Filter Cosmopedia subsets using medical quality filter. Looks for high quality medically relevant text.')

    parser.add_argument('--model_path', default='/weka/home-griffin/weights/quality-filter/roberta-large-25k/checkpoint-12000')
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--out_dir', default='/weka/home-griffin/clinical_pile/cosmopedia')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    hf_out_dir = os.path.join(args.out_dir, 'dataset_hf')

    pipe = pipeline('text-classification', model=args.model_path, torch_dtype='auto', device_map='auto')

    all_quality_data = []

    for subset in SUBSETS:
        print(f'Loading {subset}')
        dataset = load_dataset('HuggingFaceTB/cosmopedia', subset, split='train')

        keep_idxs = []
        keep_labels = []
        idx = 0
        for out in tqdm(pipe(KeyDataset(dataset, 'text'), batch_size=128, truncation='only_first')):
            if out['label'] in QUALITY_LABELS:
                keep_labels.append(out['label'])
                keep_idxs.append(idx)
            idx += 1
        
        print(f'{len(keep_labels)} / {len(dataset)} from {subset} identified as quality.')

        quality_data = dataset.select(keep_idxs)
        quality_data = quality_data.add_column('quality', keep_labels)
        quality_data = quality_data.add_column('subset', [subset for _ in range(len(quality_data))])

        all_quality_data.append(quality_data)
    
    all_quality_data = concatenate_datasets(all_quality_data)

    print(f'Saving {len(all_quality_data)} examples to {hf_out_dir}')
    all_quality_data.save_to_disk(hf_out_dir)
