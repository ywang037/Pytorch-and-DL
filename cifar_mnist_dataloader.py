import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision import datasets
import torch.utils.data as data

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# batch_size = 100
# datapath = '..\local_arxiv\cifar10'

def data_cifar(path, batch_size=100):
    """
    returns training data loader and test data loader
    """
    # no brainer normalization used in the pytorch tutorial
    mean_0 = (0.5, 0.5, 0.5)
    std_0 = (0.5, 0.5, 0.5)

    # alternative normilzation
    mean_1 = (0.4914, 0.4822, 0.4465)
    std_1 = (0.2023, 0.1994, 0.2010)

    # configure tranform for training data
    # standard transform used in the pytorch tutorial 
    transform_train_0 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_0,std_0),
    ])

    # configure transform for test data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_0,std_0),
    ])
    
    '''
    # Alternative transform
    # enhanced transform, random crop and flip is optional
    transform_train_1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_1,std_1),
    ])

    # alternative, only random crop is used
    transform_train_2 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean_1,std_1),
    ])

    # configure transform for test data
    transform_test_1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_1,std_1),
    ])
    '''
    # setup the CIFAR10 training dataset
    data_train = datasets.CIFAR10(root=path, train=True, download=False, transform=transform_train_0)
    loader_train = data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

    # setup the CIFAR10 test dataset
    data_test = datasets.CIFAR10(root=path, train=False, download=False, transform=transform_test_0)
    loader_test = data.DataLoader(data_test, batch_size=100, shuffle=False)

    return loader_train, loader_test

def data_mnist(path,batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # setup the MNIST training dataset
    data_train = datasets.MNIST(root=path, train=True, download=False, transform=transform)
    loader_train = data.DataLoader(data_train, batch_size=batch_size, shuffle=True) 
    
    # setup the MNIST training dataset
    data_test = datasets.MNIST(root=path, train=False, download=False, transform=transform)
    loader_test = data.DataLoader(data_test, batch_size=100, shuffle=False)
    return loader_train,loader_test