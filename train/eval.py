from collections import defaultdict
import torch
import argparse
import os
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from ckpt_to_hf import load_model

from vllm import LLM, SamplingParams
from tqdm import tqdm
import string

EVAL_DATASETS = {
    'multimedqa': '/weka/home-griffin/clinical_instructions/multimedqa/dataset_hf'
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert a SafeTensors checkpoint into a HuggingFace dataset so it can be loaded with "from_pretrained".')
    
    parser.add_argument('--model_name', default='Qwen/Qwen1.5-0.5B')
    
    parser.add_argument('--weight_dir', default='/weka/home-griffin/weights/finetune/Qwen/Qwen1.5-0.5B')
    parser.add_argument('--experiment', default='baseline_0.5B')
    parser.add_argument('-eval_pretrained', default=False, action='store_true')
    parser.add_argument('--ckpt', type=int, default=60000)

    parser.add_argument('--dataset', default='multimedqa')

    args = parser.parse_args()

    if args.eval_pretrained:  # Ignore experiment, weight_dir, ckpt information (we are loading from Hub)
        model_dir = args.model_name
    else:
        model_dir = os.path.join(args.weight_dir, args.experiment, f'hf_{args.ckpt}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
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
    ).eval().to('cuda')

    # sampling_params = SamplingParams(temperature=0.0)
    # model_dir = 'Qwen/Qwen1.5-0.5B'
    # llm = LLM(model=model_dir, dtype=torch.bfloat16)

    out_fn = os.path.join(model_dir, f'{args.dataset}_results.csv')

    test = load_from_disk(EVAL_DATASETS[args.dataset])['test']

    accuracy = defaultdict(lambda: [0.0, 0.0, []])

    pred_label_dist = {}

    prev_source = None

    for row in tqdm(test):
        source, prompt, completion, num_options = row['source'], row['prompt'], row['completion'], row['num_options']

        if source not in pred_label_dist:
            pred_label_dist[source] = [0 for _ in range(num_options)]
        
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.int64).to(model.device).unsqueeze(0)
    
        letter_options = list(string.ascii_uppercase[:num_options])
        assert completion in letter_options
        label_ids = tokenizer.convert_tokens_to_ids(letter_options)
        ground_truth = tokenizer.convert_tokens_to_ids([completion])[0]
        assert ground_truth in label_ids
        ground_truth_idx = label_ids.index(ground_truth)

        with torch.no_grad():
            output = model(input_ids=input_ids)
        logits = output.logits

        # Last position in sequence
        pred_probs = torch.softmax(logits[0, -1, label_ids], dim=0)
        true_likelihood = float(pred_probs[ground_truth_idx])
        pred_answer_idx = int(torch.argmax(pred_probs))

        pred_label_dist[source][pred_answer_idx] += 1

        if pred_answer_idx == ground_truth_idx:
            accuracy[source][0] += 1.
        accuracy[source][1] += 1.
        accuracy[source][2].append(true_likelihood)

        if prev_source is not None and prev_source != source:
            acc = round(accuracy[prev_source][0] / accuracy[prev_source][1] * 100, 2)
            print(f'{prev_source}: Accuracy={acc}%. Likelihood={round(sum(accuracy[prev_source][2]) / len(accuracy[prev_source][2]), 2)}')

        prev_source = source

    # encodings = tokenizer(prompts, padding=True, max_length=4096)

    # outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    # for source, output, label in zip(sources, outputs, labels):
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text.strip()
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    #     assert label in {'A', 'B', 'C', 'D'}
    #     # assert generated_text in {'A', 'B', 'C', 'D'}

    #     if generated_text.lower() == label.lower():
    #         accuracy[source][0] += 1
    #     accuracy[source][1] += 1

    print('Drumroll...')
    for source, acc in accuracy.items():
        print(f'{source}: Accuracy={round(acc[0] / acc[1] * 100, 2)}%. Likelihood={round(sum(acc[2]) / len(acc[2]), 2)}')
        print('Prediction Label Bias Analysis')
        for i in range(len(pred_label_dist[source])):
            ct = pred_label_dist[source][i]
            letter = string.ascii_uppercase[i]
            print(f'\t-{letter} predicted {ct} times.')
