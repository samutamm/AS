
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_train_test_loaders(batch_size=64, label1 = 1, label2 = 3, binary=True):
    loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, 
                              transform=transforms.Compose(
                                  [transforms.ToTensor(),
                                  transforms.Normalize((0.1307,),(0.3081,))]
                              )), 
        batch_size=batch_size, 
        shuffle=True
    )
    
    if binary:
        labels = loader.dataset.train_labels
        mask = (labels == label1) + (labels == label2) > 0

        loader.dataset.train_data = loader.dataset.train_data[mask]
        loader.dataset.train_labels = loader.dataset.train_labels[mask]

        labels = torch.where(loader.dataset.train_labels == label1, torch.ones(1), -torch.ones(1))
        loader.dataset.train_labels = labels

    full_dataset = loader.dataset.train_data
    N = full_dataset.size()[0]
    train_size = int(0.7 * N)
    val_size = int(0.2 * N)
    test_size = N - train_size - val_size
    train_indices, val_indices, test_indices = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        loader.dataset, 
        batch_size=batch_size, 
        sampler=SubsetRandomSampler(train_indices.indices)
    )
    
    val_loader = torch.utils.data.DataLoader(
        loader.dataset, 
        batch_size=batch_size, 
        sampler=SubsetRandomSampler(val_indices.indices)
    )

    test_loader = torch.utils.data.DataLoader(
        loader.dataset, 
        batch_size=batch_size, 
        sampler=SubsetRandomSampler(test_indices.indices)
    )
    
    return train_loader, val_loader, test_loader, train_size, val_size, test_size