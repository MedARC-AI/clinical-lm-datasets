from collections import Counter
import argparse
import os
from datasets import load_from_disk, concatenate_datasets
from glob import glob
import numpy as np
np.random.seed(4692)
import regex as re
from tqdm import tqdm
import string
from nltk import sent_tokenize
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer


EVAL_DATASETS = {
    'multimedqa': '/weka/home-griffin/clinical_instructions/multimedqa/dataset_hf',
    'multimedqa_cot': '/weka/home-griffin/clinical_instructions/multimedqa/dataset_cot_hf_artificial',
}


FEWSHOT_REPLACEMENTS = {
    'medqa': 'medmcqa',
    'mmlu': 'medmcqa'
}


def classification_report(source, labels, preds, letter_options):
    confusion = np.zeros(shape=(len(letter_options), len(letter_options)))
    for a, b in zip(labels, preds):
        confusion[letter_options.index(a), letter_options.index(b)] += 1
    n = len(preds)
    num_correct = sum([1 if a == b else 0 for a, b in zip(preds, labels)])
    acc = round(100 * num_correct / n, 3)
    out_lines = [
        f'{model_dir} for {source}',
        f'Accuracy: {acc}%',
        pretty_print_2d_array(confusion)
    ]

    out_str = '\n\n'.join(out_lines)
    return out_str


def extract_answer_logit(logprobs, letter_options, tokenizer):
    candidate_logits = []
    for label in letter_options:
        try:
            candidate_logits.append(logprobs[tokenizer.convert_tokens_to_ids(label)])
        except:
            # If an option is not in the first 1000, set its logit to -100
            print(f'\n\WARNING! {label} not found. Artificially adding log prob of -100.\n\n')
            candidate_logits.append(-100)
    candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
    probs = (
        torch.nn.functional.softmax(
            candidate_logits,
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )
    return letter_options[int(np.argmax(probs))]


def remove_dup_sents(cot):
    sents = sent_tokenize(cot)
    filt_sents = []
    for sent in sents:
        if sent in filt_sents:
            continue
    
        filt_sents.append(sent)
    return ' '.join(filt_sents)


def pretty_print_2d_array(array_2d):
    # Finding the maximum width of the numbers in the array for alignment
    max_width = max(len(str(item)) for row in array_2d for item in row)
    
    # Building the string representation with gridlines
    row_lines = []
    for row in array_2d:
        row_str = '| ' + ' | '.join(f"{item:>{max_width}}" for item in row) + ' |'
        row_lines.append(row_str)
    
    # Joining all rows with a newline character
    pretty_array_str = '\n'.join(row_lines)
    
    return pretty_array_str


def form_cot_prompt(args, row, fewshot_examples):
    prompt = row['prompt']
    sampled_fewshot = []
    if args.fewshot_n == 0:
        input = prompt
    else:
        # Add few-shot
        sampled_fewshot = list(np.random.choice(fewshot_examples, args.fewshot_n))
        np.random.shuffle(sampled_fewshot)
        input = '\n\n**********\n\n'.join(sampled_fewshot + [prompt])

    input += '\n\n# EXPLANATION\n'

    return input


def load_model_info(args):
    if 'weka' not in args.model_path:
        model_dir = args.model_path
        save_dir = os.path.join('/weka/home-griffin/weights', 'base', args.model_path)
    elif args.ckpt == -1:
        model_dir = os.path.join(args.model_path, 'hf_final')
        save_dir = model_dir
    else:
        ckpt_name = str(args.ckpt)
        model_dir = os.path.join(args.model_path, f'hf_{ckpt_name}')
        save_dir = model_dir
    return save_dir, model_dir


def get_fewshot_examples(args, source, dataset):
    fewshot_examples = []

    if args.fewshot_n > 0:
        fewshot_pool = []
        fewshot_source = FEWSHOT_REPLACEMENTS.get(source, source)

        for split in ['train', 'validation']:
            if split in dataset:
                fewshot_pool.append(dataset[split])
        fewshot_pool = concatenate_datasets(fewshot_pool)

        if fewshot_source == 'mmlu':
            fewshot_pool = fewshot_pool.filter(lambda row: fewshot_source in row['source']).shuffle()
        else:
            fewshot_pool = fewshot_pool.filter(lambda row: fewshot_source == row['source']).shuffle()

        # Available for fewshot sampling
        print(f'{len(fewshot_pool)} in the pool for few-shot exemplars (Train + Validation)')
        fewshot_pool = fewshot_pool.filter(lambda x: len(x['explanation']) > 0)

        fewshot_examples = [
            row['prompt'].strip() + '\n\n' + row['completion'].strip() for row in fewshot_pool
        ]
    return fewshot_examples


def run_tests(args, test, fewshot_examples, llm, gen_params, logit_params, tokenizer):
    all_labels = test['label']

    gen_predictions = []
    gen_labels = []
    
    if args.cot:
        print('Forming fewshot CoT prompts...')
        gen_inputs = [form_cot_prompt(args, row, fewshot_examples) for row in test]    
        print('Generating predictions...')
        # Generate Predictions
        gen_outputs = llm.generate(gen_inputs, gen_params)
        print('Done generating predictions...')

        # If we can't extract answer we add to this
        logit_inputs = []
        logit_labels = []

        # Print the outputs.
        for label, output in zip(all_labels, gen_outputs):
            prompt = output.prompt
            new_text = output.outputs[0].text

            if '**********' in new_text:
                new_text = new_text.split('**********')[0]

            if num_options == 3:
                pattern = '# ANSWER\s+([A-C])'
            elif num_options == 4:
                pattern = '# ANSWER\s+([A-D])'
            else:
                raise Exception(f'Create new regex (lazy) for {num_options}')

            pred_label = re.search(pattern, new_text)

            if pred_label is None:
                print(new_text)
                gen_cot = remove_dup_sents(new_text).strip()
                answer_prompt = '\n\n# ANSWER\n'
                updated_inputs = prompt + gen_cot + answer_prompt    
                logit_inputs.append(updated_inputs)
                logit_labels.append(label)
            else:
                pred_label = pred_label.group(1)
                gen_labels.append(label)
                gen_predictions.append(pred_label)
    else:
        logit_inputs = test['prompt']
        logit_labels = test['label']

    assert len(logit_inputs) == len(logit_labels)
    logit_predictions = []
    if len(logit_inputs) > 0:
        if args.cot:
            print(f'Unable to extract answer from {len(logit_inputs)}. Will extract logits from them.')
        print('Extracting logits...')
        logit_outputs = llm.generate(logit_inputs, logit_params)
        for output in logit_outputs:
            logprobs = output.outputs[0].logprobs[0]
            logit_pred = extract_answer_logit(logprobs, letter_options, tokenizer)
            logit_predictions.append(logit_pred)

    all_labels = gen_labels + logit_labels
    all_preds = gen_predictions + logit_predictions
    return all_labels, all_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Chain of Thought Evaluations on MultimedQA')

    parser.add_argument('--model_path', default='/weka/home-griffin/weights/finetune/Qwen/Qwen1.5-7B')

    parser.add_argument('--sources', default='all')
    parser.add_argument('--fewshot_n', type=int, default=0)
    parser.add_argument('--ckpt', type=int, default=-1)  # -1 means take the last checkpoint
    parser.add_argument('-cot', default=False, action='store_true')

    parser.add_argument('--dataset', default='multimedqa', choices=['multimedqa'])  # Only Supported Dataset right now
    parser.add_argument('-overwrite', default=False, action='store_true')

    args = parser.parse_args()

    save_dir, model_dir = load_model_info(args)

    # VLLM Engine Stuff
    print(f'Loading tokenizer from {args.model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=model_dir, dtype=torch.bfloat16)
    gen_params = SamplingParams(temperature=0, max_tokens=1024)
    logit_params = SamplingParams(temperature=1, max_tokens=1, logprobs=100)

    # DATA stuff
    data_path = args.dataset
    if args.cot:
        data_path += '_cot'
    dataset = load_from_disk(EVAL_DATASETS[data_path])
    test = dataset['test']
    print(Counter(test['source']))

    if args.sources == 'all':
        sources = list(sorted(list(set(test['source']))))
        sources = [x for x in sources if 'mmlu' not in x] + ['mmlu']
        sources = list(sorted(sources))
    else:
        sources = list(args.sources.split(','))

    for source in sources:
        print(f'Starting with {source}')
        prefix = 'cot' if args.cot else 'll'
        out_fn = os.path.join(save_dir, f'{prefix}_results_{source}_{args.fewshot_n}_shot.txt')

        if os.path.exists(out_fn) and not args.overwrite:
            print(f'{out_fn} already exists. Remove with "rm -rf {out_fn}" or run with -overwrite.')
            with open(out_fn, 'r') as fd:
                print(fd.read())
            continue
        elif os.path.exists(out_fn):
            print(f'Will be over-writing previous results ({out_fn}) below...')
            with open(out_fn, 'r') as fd:
                print(fd.read())
        else:
            print(f'Generating predictions for {source} and saving to {out_fn}')

        fewshot_examples = get_fewshot_examples(args, source, dataset)

        if source == 'mmlu':
            source_test = test.filter(lambda row: 'mmlu' in row['source'], load_from_cache_file=False)
        else:
            source_test = test.filter(lambda row: source == row['source'], load_from_cache_file=False)

        # Inputs for VLLM
        num_options = source_test['num_options'][0]
        letter_options = list(string.ascii_uppercase[:num_options])

        source_labels, source_preds = run_tests(args, source_test, fewshot_examples, llm, gen_params, logit_params, tokenizer)

        full_report = classification_report(source, source_labels, source_preds, letter_options=letter_options)

        print(source)
        print(full_report)
        os.makedirs(save_dir, exist_ok=True)

        with open(out_fn, 'w') as fd:
            fd.write(full_report)
