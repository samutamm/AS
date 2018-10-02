
import numpy as np

def pad_to_32(X, y):
    N = X.shape[0]
    smaller = np.floor(N / 32)
    repeat = 32 - (N - int(smaller * 32))
    repeated = np.tile(X[-1],(repeat,1))
    #import pdb; pdb.set_trace()
    new_X = np.vstack((X,repeated))
    new_y = np.concatenate((y,np.repeat(y[-1],repeat)))
    return new_X, new_y

def get_patches(N):
    idx = np.arange(N)
    np.random.shuffle(idx)
    return np.split(idx, int(N/32))