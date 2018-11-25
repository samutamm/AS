
import numpy as np
import torch

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

def get_minibatches(loader, device):
    for data, target in loader:
        normalize = torch.nn.BatchNorm2d(1)
        data = normalize(data)
        
        data = torch.squeeze(data)
        target = target.cuda(async=True)
        data = data.cuda(async=True)
        
        batch_n = data.size()[0]
        X = data.view(batch_n, -1)
        ones = torch.ones((X.size()[0], 1), device=device)
        X = torch.cat((X, ones), 1)
        
        y_onehot = torch.zeros((target.size()[0], 10), device=device)
        y_onehot.zero_()
        y_onehot.scatter_(1, target.view(-1,1), 1)
        
        X = torch.autograd.Variable(X)
        y = torch.autograd.Variable(y_onehot)
        yield X, y.long()