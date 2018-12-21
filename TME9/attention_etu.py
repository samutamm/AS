import sys
sys.path.insert(0, "/users/Enseignants/piwowarski/.local/lib/python3.6/site-packages")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import spacy
import torchtext.datasets as datasets
import torchtext.data as ttdata
import math

import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR="/home/samutamm/Documents/Study/Sorbonne/AS/TME/data"

DATASET_DIR="%s/data" % DATA_DIR
VECTORS_DIR="%s/vectors" % DATA_DIR
#DATA_DIR=Path("/users/Enseignants/piwowarski")
#DATASET_DIR=str(DATA_DIR.joinpath("datasets"))
#VECTORS_DIR=str(DATA_DIR.joinpath("vectors"))

wordsemb_dim = 100
wordsemb = torchtext.vocab.GloVe("6B", dim=wordsemb_dim, cache=VECTORS_DIR)
TOKENIZER="spacy"


class Data:
    def __init__(self, batch_first=False):
        self.TEXT=ttdata.Field(lower=True,include_lengths=False,batch_first=batch_first,tokenize=TOKENIZER)
        self.LABEL = ttdata.Field(sequential=False, is_target=True)

    def __repr__(self):
        return "{}({},{},{})".format(self.__class__.__name__, len(self.train), len(self.val), len(self.test))
    
    @property
    def wordemb_dim(self):
        return self.wordemb.weight.shape[1]
    
    @property
    def target_dim(self):
        return len(self.LABEL.vocab) - 1

    def batches(self, batch_size):
        # Batch

        train_iter, val_iter, test_iter = \
          ttdata.BucketIterator.splits((self.train, self.val, self.test), batch_size=batch_size, device=device)

        return  {
            "train": train_iter,
            "val": val_iter,
            "test": test_iter
        }
    
class SST(Data):
    def __init__(self, wordsemb=wordsemb, **kwargs):
        # text
        super().__init__(**kwargs)

        # make splits for data
        # Build the vocabularies
        # for labels, we use special_first to False so <unk> is last
        # (to discard it)
        self.train, self.val, self.test = datasets.sst.SST.splits(self.TEXT, self.LABEL, root=DATASET_DIR)

        self.TEXT.build_vocab(self.train, vectors=wordsemb)
        self.LABEL.build_vocab(self.train, specials_first=False)
        self.wordemb = nn.Embedding.from_pretrained(self.TEXT.vocab.vectors).to(device)




class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AttentionModel(nn.Module):

    def __init__(self, embedding_size, classes, layers=None):
        super(AttentionModel, self).__init__()
        self.f_theta = nn.Linear(embedding_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.decoder = nn.Linear(embedding_size, classes)

    def forward(self, x):

        l1 = self.f_theta(x)
        probas = self.softmax(l1)
        y_embed = (probas * x).sum(dim=1)
        return self.decoder(y_embed)


class AttentionModel2(nn.Module):

    def __init__(self, embedding_size, classes, layers=None):
        super(AttentionModel2, self).__init__()
        self.f_aij = nn.Linear(embedding_size, embedding_size)

        self.f_theta = nn.Linear(embedding_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.decoder = nn.Linear(embedding_size, classes)

    def forward(self, x):
        pre_aij = torch.mm(self.f_aij(x), x)
        aij = self.softmax(pre_aij)

        import pdb; pdb.set_trace()
        W_l1 = (aij * x).sum(dim=1)
        l1 = self.f_theta(x)
        probas = self.softmax(l1)
        y_embed = (probas * x).sum(dim=1)
        return self.decoder(y_embed)



def model1(data):
    model = AttentionModel(wordsemb_dim, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    ### Utilisation
    sets = data.batches(batch_size)
    for epoch in range(5):
        losses = []
        for batch in sets["train"]:
            x=data.wordemb(batch.text)
            output = model(x)

            loss = criterion(output, batch.label)
            losses.append(loss.cpu().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(np.mean(losses))


def model2(data):
    model = AttentionModel2(wordsemb_dim, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    ### Utilisation
    sets = data.batches(batch_size)
    for epoch in range(5):
        losses = []
        for batch in sets["train"]:
            x = data.wordemb(batch.text)
            output = model(x)

            loss = criterion(output, batch.label)
            losses.append(loss.cpu().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(np.mean(losses))

batch_size = 32
data = SST(batch_first=True)
model2(data)
