# define our model to be used for CIFAR10 baseline experiments

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
from torchvision import transforms
from torchvision import datasets

class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar,self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self,x):
        x=self.conv_layer(x)
        x=x.view(-1, 16 * 5 * 5)
        logits=self.fc_layer(x)
        return F.log_softmax(logits,dim=1)


