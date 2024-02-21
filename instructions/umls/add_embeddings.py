import os

import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


BASE_DIR = '/weka/home-griffin/clinical_pile/umls'
SAP_BERT = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'


def embed_concept_spans(model, tokenizer, texts, batch_size=16384, verbose=True):
    all_reps = []
    batch_starts = np.arange(0, len(texts), batch_size)
    num_batches = len(batch_starts)
    batch_ct = 0
    for i in batch_starts:
        batch_ct += 1
        toks = tokenizer.batch_encode_plus(
            texts[i:i + batch_size], padding='max_length', max_length=25, truncation=True, return_tensors='pt')
        toks_cuda = {k: v.to(model.device) for k, v in toks.items()}
        with torch.no_grad():
            output = model(**toks_cuda)
            cls_rep = output[0][:, 0, :]
            all_reps.append(cls_rep.cpu().detach().numpy())
        if verbose:
            print(f'{batch_ct}/{num_batches}')
    return np.concatenate(all_reps, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add SapBERT embeddings for all CUI descriptions')
    
    parser.add_argument('--sources', default='all', choices=['all', 'level_0'])
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(SAP_BERT)
    model = AutoModel.from_pretrained(SAP_BERT, dtype='auto').eval().to(args.device)

    in_fn = os.path.join(BASE_DIR, args.sources, 'cui.csv')
    print(f'Reading in CUIs from {in_fn}')

    cui_df = pd.read_csv(in_fn)
    print(f'Loaded {len(cui_df)} CUIs from {in_fn}')

    names = cui_df['name'].tolist()

    embed_concept_spans(model, tokenizer, names)