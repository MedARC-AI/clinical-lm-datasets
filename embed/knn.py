

import argparse

import faiss
   
BASE_DIR = '/weka/home-griffin/clinical_pile/v1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--index_path', default='/weka/home-griffin/clinical_pile/v1/knn.index')
    parser.add_argument('--index_id_path', default='/weka/home-griffin/clinical_pile/v1/ids.txt')

    parser.add_argument('--k', default=10, type=int)

    args = parser.parse_args()

    print(f'Loading index at ')
    index = faiss.read_index(args.index_path)
    assert index.is_trained

    import numpy as np
    query = np.zeros([1, 4096])
    D, I = index.search(query, args.k)
    print(D)
    print(I)
