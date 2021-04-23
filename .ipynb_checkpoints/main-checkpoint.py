import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
from torchvision import transforms
from torchvision import datasets

from models import CNNCifar
from data_preparation import data_setup

class HyperParam()：
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 100
    datapath = '..\local_arxiv\cifar10'