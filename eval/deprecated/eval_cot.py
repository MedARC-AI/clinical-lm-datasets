from collections import defaultdict, Counter
import torch
import argparse
import os
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import gather_object

from glob import glob
import numpy as np
np.random.seed(4692)
import regex as re
from tqdm import tqdm
import string
from nltk import sent_tokenize
import time


EVAL_DATASETS = {
    'multimedqa': '/weka/home-griffin/clinical_instructions/multimedqa/dataset_cot_hf_artificial',
}


FEWSHOT_REPLACEMENTS = {
    'medqa': 'medmcqa',
    'mmlu': 'medmcqa'
}


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Chain of Thought Evaluations on MultimedQA')
    
    parser.add_argument('--model_name', default='Qwen/Qwen1.5-7B')
    
    parser.add_argument('--weight_dir', default='/weka/home-griffin/weights/finetune/Qwen/Qwen1.5-7B')
    parser.add_argument('--experiment', default='qwen_7b_base_pubmedqa_v2')
    parser.add_argument('-eval_pretrained', default=False, action='store_true')
    parser.add_argument('--source', default='pubmedqa_labeled')
    parser.add_argument('--fewshot_n', type=int, default=0)
    parser.add_argument('--ckpt', type=int, default=-1)  # -1 means take the last checkpoint

    parser.add_argument('--dataset', default='multimedqa', choices=['multimedqa'])  # Only Supported Dataset right now
    parser.add_argument('-overwrite', default=False, action='store_true')

    args = parser.parse_args()

    if args.eval_pretrained:
        model_dir = args.model_name
        save_dir = os.path.join('/weka/home-griffin/weights', 'base', args.model_name)
    elif args.ckpt == -1:
        print(f'Searching for latest checkpoint in {args.ckpt}...')
        pattern = os.path.join(args.weight_dir, args.experiment, '*.safetensors')
        all_ckpts = list(glob(pattern))

        if len(all_ckpts) == 0:
            print(f'No checkpoints found in {os.path.join(args.weight_dir, args.experiment)}.')
        args.ckpt = max([int(re.search('model_state_dict_(\d+)', fn).group(1)) for fn in all_ckpts])
        print(f'Found checkpoint --> {args.ckpt}')
        ckpt_name = 'final'
        model_dir = os.path.join(args.weight_dir, args.experiment, f'hf_{ckpt_name}')
        save_dir = model_dir
    else:
        ckpt_name = str(args.ckpt)
        model_dir = os.path.join(args.weight_dir, args.experiment, f'hf_{ckpt_name}')
        save_dir = model_dir

    out_fn = os.path.join(save_dir, f'cot_results_{args.source}_{args.fewshot_n}_shot.txt')

    if os.path.exists(out_fn) and not args.overwrite:
        print(f'{out_fn} already exists. Remove with "rm -rf {out_fn}" or run with -overwrite.')
        with open(out_fn, 'r') as fd:
            print(fd.read())
        exit(0)
    elif os.path.exists(out_fn):
        print(f'Will be over-writing previous results ({out_fn}) below...')
        with open(out_fn, 'r') as fd:
            print(fd.read())

    print(f'Loading tokenizer from {args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # if not args.eval_pretrained and not os.path.exists(model_dir):
    #     ckpt = os.path.join(args.weight_dir, args.experiment, f'model_state_dict_{args.ckpt}.safetensors')
    #     print(f'Converting {ckpt} to HF dataset...')
    #     model = load_model(args.model_name, ckpt=ckpt)
    #     print(f'Saving model and tokenizer to {model_dir}...')
    #     model.save_pretrained(model_dir)
    #     tokenizer.save_pretrained(model_dir)

    dataset = load_from_disk(EVAL_DATASETS[args.dataset])
    test = dataset['test']

    if args.fewshot_n > 0:
        fewshot_pool = []
        fewshot_source = FEWSHOT_REPLACEMENTS.get(args.source, args.source)

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

    if args.source == 'mmlu':
        test = test.filter(lambda row: 'mmlu' in row['source'], load_from_cache_file=False).shuffle()
    else:
        test = test.filter(lambda row: args.source == row['source'], load_from_cache_file=False).shuffle()
    print(Counter(test['source']))

    prompts_all = test['prompt']
    labels = test['label']
    num_options = test['num_options'][0]
    letter_options = list(string.ascii_uppercase[:num_options])
    malformed = 0

    accelerator = Accelerator()

    print(f'Loading model from {model_dir}')
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        device_map={'': accelerator.process_index},
    ).eval()  # .to('cuda')

    accelerator.wait_for_everyone()
    start = time.time()

    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(list(test)) as rows:
        # store output of generations in dict
        results=dict(predicted=[], label=[])

        # have each GPU do inference, prompt by prompt
        for row in rows:
            prompt, label = row['prompt'], row['label']
            sampled_fewshot = []
            if args.fewshot_n == 0:
                input = prompt
            else:
                # Add few-shot
                sampled_fewshot = list(np.random.choice(fewshot_examples, args.fewshot_n))
                np.random.shuffle(sampled_fewshot)
                input = '\n\n**********\n\n'.join(sampled_fewshot + [prompt])

            input += '\n\n# EXPLANATION\n'

            prompt_tokenized=tokenizer(input, return_tensors='pt').to('cuda')
            output_ids = model.generate(**prompt_tokenized, pad_token_id=tokenizer.pad_token_id, min_new_tokens=8, max_new_tokens=256)[0]
            new_ids = output_ids[len(prompt_tokenized['input_ids'][0]):]

            # remove prompt from output
            new_text = tokenizer.decode(new_ids, skip_special_tokens=True)

            if '**********' in new_text:
                new_text = new_text.split('**********')[0]

            if num_options == 3:
                pattern = '# ANSWER\s+([A-C])'
            elif num_options == 4:
                pattern = '# ANSWER\s+([A-D])'
            else:
                raise Exception(f'Create new regex (lazy) for {num_options}')

            pred_label = re.search(pattern, new_text)
            # label_id = letter_options.index(row['label'])
            if pred_label is None:
                # if '# ANSWER' in new_text:
                #     print(new_text)
                #     print('Tried to give answer but not in letter format. Fix this ultimately.')
                gen_cot = remove_dup_sents(new_text).strip()
                answer_prompt = '\n\n# ANSWER\n'
                updated_inputs = input + gen_cot + answer_prompt
                # print(f'Malformed output...Re-trying with this prompt...\n{updated_inputs}')

                input_ids = torch.tensor(tokenizer.encode(updated_inputs), dtype=torch.int64).unsqueeze(0).to('cuda')
                attention_mask = torch.ones_like(input_ids).to('cuda')

                with torch.no_grad():
                    output = model(input_ids=input_ids, attention_mask=attention_mask)

                option_vocab_ids = tokenizer.convert_tokens_to_ids(letter_options)
                logits = output.logits[0, -1]
                pred_probs = torch.softmax(logits[option_vocab_ids], dim=0)
                pred_answer_idx = int(torch.argmax(pred_probs))
                pred_label = letter_options[pred_answer_idx]
            else:
                pred_label = pred_label.group(1)

            # store outputs and number of tokens in result{}
            results['predicted'].append(pred_label)
            results['label'].append(label)

            n = len(results['predicted'])
            now = time.time()
            minutes = round((now - start) / 60.0, 2)
            num_correct = sum([1 if a == b else 0 for a, b in zip(results['predicted'], results['label'])])
            acc = round(100 * num_correct / n, 2)
            print(f'{accelerator.process_index}: {n} / {len(rows)} done in {minutes} minutes --> {acc}%')

        results=[results] # transform to list, otherwise gather_object() will not collect correctly

    # collect results from all the GPUs
    results_gathered=gather_object(results)

    if accelerator.is_main_process:
        preds = []
        labels = []
        for r in results_gathered:
            labels += r['label']
            preds += r['predicted']
        
        n = len(preds)
        num_correct = sum([1 if a == b else 0 for a, b in zip(preds, labels)])
        acc = round(100 * num_correct / n, 3)
        print(acc)

        confusion = np.zeros(shape=(num_options, num_options))

        for r, c in zip(labels, preds):
            confusion[letter_options.index(r), letter_options.index(c)] += 1

        out_lines = [
            f'{model_dir} for {args.source}',
            f'Accuracy: {acc}%',
            pretty_print_2d_array(confusion)
        ]

        out_str = '\n\n'.join(out_lines)
        print(out_str)
        os.makedirs(save_dir, exist_ok=True)

        print(f'Saving to {out_str}')
        with open(out_fn, 'w') as fd:
            fd.write(out_str)
