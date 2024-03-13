import os
from glob import glob


base_dir = '/weka/home-griffin/clinical_pile/v1/embeddings'


if __name__ == '__main__':
    chunk_dirs = sorted(os.listdir(base_dir), key=lambda x: int(x.split('-')[0]))

    for chunk in chunk_dirs:
        subdirs = os.listdir(os.path.join(base_dir, chunk))
        if 'dataset_info.json' in subdirs:
            print(f'Chunk {chunk} -> finished')
        else:
            print(f'Chunk {chunk} -> {len(subdirs)} / {100}')
