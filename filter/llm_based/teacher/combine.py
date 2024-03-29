from glob import glob
from p_tqdm import p_uimap
import os
import ujson
from datasets import Dataset, load_from_disk
import pandas as pd
from tqdm import tqdm


def load(fn):
    with open(fn, 'r') as fd:
        return ujson.load(fd)

def concat(arr):
    return pd.concat(arr)


if __name__ == '__main__':
    dimension = 'topic'

    dir = f'/weka/home-griffin/clinical_pile/v1/dataset_hf_1mn_sample_llm_{dimension}_scores'
    fns = list(glob(os.path.join(dir, '*.json')))
    
    out_dir = os.path.join(dir, 'hf')
    df = []
    if os.path.exists(out_dir):
        # print(f'{out_dir} already exists. Remove first.')
        # exit(0)
        df.append(pd.DataFrame(load_from_disk(out_dir)).reset_index(drop=True))
        out_dir += '_v2'

    df.append(pd.DataFrame(list(p_uimap(load, fns))).reset_index(drop=True))
    print(len(df[0]), len(df[-1]))
    df = concat(df)
    print(len(df))
    df = df.drop_duplicates(subset=['uuid'])
    print(len(df))
    df.reset_index(drop=True, inplace=True)
    # df = df.loc[~df.index.duplicated(),:].copy()
    dataset = Dataset.from_pandas(df)
    print(f'Saving {len(dataset)} examples to {out_dir}')
    dataset.save_to_disk(out_dir)

    # print(f'Now removing {len(fns)}')
    # for fn in tqdm(fns):
    #     os.remove(fn)
