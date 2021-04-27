# this is the python modules that contains the toy python/pytorch classes when I try to do hands-on practice of model creation and training

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision import datasets
import torch.utils.data as data


class MySimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
        nn.Linear(3072,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10))
        
    def forward(self,x):
        logits = self.layer_stack(x.reshape(1,-1).squeeze())
        return logits
    
class MyLinearModel(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in,dim_out)
        
    def forward(self,x):
        return self.linear(x)

