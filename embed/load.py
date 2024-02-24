import h5py
import numpy as np


if __name__ == '__main__':
    h5f = h5py.File('embeddings.h5','r')
    embeddings = np.array(h5f.get('array'))
