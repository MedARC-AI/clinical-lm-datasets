from collections import defaultdict, Counter
import torch
import argparse
import os
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from ckpt_to_hf import load_model

from glob import glob
from torch.nn.utils.rnn import pad_sequence
import numpy as np
np.random.seed(4692)
import regex as re
from tqdm import tqdm
import string


EVAL_DATASETS = {
    'multimedqa': '/weka/home-griffin/clinical_instructions/multimedqa/dataset_hf',
    'instruction_pile': '/weka/home-griffin/clinical_instructions/v1/dataset_hf'
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert a SafeTensors checkpoint into a HuggingFace dataset so it can be loaded with "from_pretrained".')

    parser.add_argument('--model_name', default='Qwen/Qwen1.5-7B')
    
    parser.add_argument('--weight_dir', default='/weka/home-griffin/weights/pretrain/qwen')
    parser.add_argument('--experiment', default='pubmed_qwen_7b_2k')
    parser.add_argument('-eval_pretrained', default=False, action='store_true')
    parser.add_argument('--fewshot_n', default=3, type=int)
    parser.add_argument('--source', default='all')
    parser.add_argument('--fewshot_split', default='train')
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--ckpt', type=int, default=-1)  # -1 means take the last checkpoint

    parser.add_argument('--dataset', default='multimedqa')
    parser.add_argument('--batch_size', default=4, type=int)

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not args.eval_pretrained and not os.path.exists(model_dir):
        ckpt = os.path.join(args.weight_dir, args.experiment, f'model_state_dict_{args.ckpt}.safetensors')
        print(f'Converting {ckpt} to HF dataset...')
        model = load_model(args.model_name, ckpt=ckpt)
        print(f'Saving model to {model_dir}...')
        model.save_pretrained(model_dir)

        tokenizer.save_pretrained(model_dir)

    print(f'Loading model from {model_dir}')
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        device_map='auto'
    ).eval()  # .to('cuda')

    sample_generate = False
    if sample_generate:
        prompt = 'According to clinical guidelines, diabetes is best treated'
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.int64).unsqueeze(0).to('cuda')

        outputs = model.generate(input_ids, min_length=64, max_length=1024)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(output_text)

    out_fn = os.path.join(save_dir, f'{args.dataset}_results.csv')

    if os.path.exists(out_fn) and not args.overwrite:
        print(f'{out_fn} exists. Exiting.')
        exit(0)
    else:
        print(f'{out_fn} exists but will be over-writing it.')

    dataset = load_from_disk(EVAL_DATASETS[args.dataset])
    test = dataset['test']
    print(Counter(test['source']))

    if args.source == 'all':
        sources = list(set(test['source']))
    else:
        sources = [args.source]
    test = test.filter(lambda row: row['source'] in set(sources))

    fewshot_samples = {s: [] for s in sources}
    if args.fewshot_n > 0:
        fewshot_data = dataset[args.fewshot_split]
        for source in sources:
            d = fewshot_data.filter(lambda row: row['source'] == source)
            fewshot_samples[source] = [x['prompt'] + x['completion'] for x in d]

    pred_label_dist = {}
    model_inputs = []

    for idx, row in tqdm(enumerate(test), total=len(test)):
        source, prompt, completion, num_options = row['source'], row['prompt'], row['completion'], row['num_options']

        if source not in pred_label_dist:
            pred_label_dist[source] = [0 for _ in range(num_options)]
        
        if len(fewshot_samples[source]) == 0:
            input = prompt
        else:
            # Add few-shot
            sampled_fewshot = list(np.random.choice(fewshot_samples[source], args.fewshot_n))
            np.random.shuffle(sampled_fewshot)
            input = '\n\n**********\n\n'.join(sampled_fewshot + [prompt])

        input_ids = torch.tensor(tokenizer.encode(input), dtype=torch.int64)
    
        letter_options = list(string.ascii_uppercase[:num_options])
        assert completion in letter_options
        label_ids = tokenizer.convert_tokens_to_ids(letter_options)
        ground_truth = tokenizer.convert_tokens_to_ids([completion])[0]
        assert ground_truth in label_ids
        ground_truth_idx = label_ids.index(ground_truth)

        model_inputs.append({
            'idx': idx,
            'source': source,
            'input_ids': input_ids,
            'last_logit_idx': len(input_ids) - 1,  # Pre-Padding where do we extract the data
            'label_ids': label_ids,
            'ground_truth_idx': ground_truth_idx,
        })

    accuracy = defaultdict(lambda: [0.0, 0.0, []])
    prev_source = None

    batch_starts = list(range(0, len(model_inputs), args.batch_size))
    for start_idx in tqdm(batch_starts):
        end_idx = min(start_idx + args.batch_size, len(model_inputs))
        batch_inputs = model_inputs[start_idx: end_idx]

        input_ids = [x['input_ids'] for x in batch_inputs]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(model.device)
        with torch.no_grad():
            output = model(input_ids=input_ids)
        logits = output.logits

        for batch_idx in range(len(logits)):
            # Last position in sequence
            mi = batch_inputs[batch_idx]
            source = mi['source']
            pred_probs = torch.softmax(logits[batch_idx, mi['last_logit_idx'], mi['label_ids']], dim=0)
            true_likelihood = float(pred_probs[mi['ground_truth_idx']])
            pred_answer_idx = int(torch.argmax(pred_probs))

            pred_label_dist[source][pred_answer_idx] += 1

            assert 0 <= mi['ground_truth_idx'] < len(pred_probs)
            if pred_answer_idx == mi['ground_truth_idx']:
                accuracy[source][0] += 1.
            accuracy[source][1] += 1.
            accuracy[source][2].append(true_likelihood)

            if prev_source is not None and prev_source != source:
                acc = round(accuracy[prev_source][0] / accuracy[prev_source][1] * 100, 2)
                print(f'{prev_source}: Accuracy={acc}%. Likelihood={round(sum(accuracy[prev_source][2]) / len(accuracy[prev_source][2]), 2)}')

            prev_source = source

    print('Drumroll...')
    out_lines = []
    for source, acc in accuracy.items():
        out_lines.append(f'{source}: Accuracy={round(acc[0] / acc[1] * 100, 2)}%. Likelihood={round(sum(acc[2]) / len(acc[2]), 2)}')
        out_lines.append('Prediction Label Bias Analysis')
        for i in range(len(pred_label_dist[source])):
            ct = pred_label_dist[source][i]
            letter = string.ascii_uppercase[i]
            out_lines.append(f'\t-{letter} predicted {ct} times.')

    print('\n'.join(out_lines))
    os.makedirs(save_dir, exist_ok=True)

    with open(out_fn, 'r') as fd:
        fd.write('\n'.join(out_lines))