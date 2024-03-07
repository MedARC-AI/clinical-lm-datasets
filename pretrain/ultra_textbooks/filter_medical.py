from datasets import load_dataset

import argparse
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Filter UltraTextbooks using medical quality filter. Looks for high quality medically relevant text.')

    parser.add_argument('--model_path', default='/weka/home-griffin/weights/quality-filter/roberta-large-25k/checkpoint-12000')
    parser.add_argument('--batch_size', default=128, type=int)

    args = parser.parse_args()

    pipe = pipeline('text-classification', model=args.model_path, torch_dtype='auto', device_map='auto')
    dataset = load_dataset('Locutusque/UltraTextbooks', split='train')
    for out in pipe(KeyDataset(dataset, 'text'), batch_size=128, truncation='only_first'):
        print(out)
