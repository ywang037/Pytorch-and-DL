# define our model to be used for CIFAR10 baseline experiments

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
from torchvision import transforms
from torchvision import datasets

# the example model used in the official CNN training tutorial of PyTorch using CIFAR10
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class CNNCifarTorch(nn.Module):
    def __init__(self):
        super(CNNCifarTorch,self).__init__()
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

# the exmaple model used in the official CNN tutorial of TensorFlow using CIFAR10
# https://www.tensorflow.org/tutorials/images/cnn
class CNNCifarTf(nn.Module):
    def __init__(self):
        super(CNNCifarTf,self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3), # output size 30*30, i.e., (32, 30 ,30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # output size 15*15, i.e., (32, 15 ,15)
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3), # output size 13*13, i.e., (64, 13 ,13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # output size 6*6, i.e., (64, 6, 6)
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3), # output size 4*4, i.e., (64, 4, 4)
            nn.ReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=1024,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=10),
        )

    def forward(self,x):
        x=self.conv_layer(x)
        x=x.view(-1,1024)
        logits = self.fc_layer(x)
        return F.log_softmax(logits,dim=1)

# WY's edition of cnn model shown on TF tutorial, with dropout layer added
class CNNCifarTfDp(nn.Module):
    def __init__(self):
        super(CNNCifarTfDp,self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3), # output size 30*30, i.e., (32, 30 ,30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # output size 15*15, i.e., (32, 15 ,15)
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3), # output size 13*13, i.e., (64, 13 ,13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # output size 6*6, i.e., (64, 6, 6)
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3), # output size 4*4, i.e., (64, 4, 4)
            nn.Dropout2d(), # this is WY's added layer to postpone overfitting, allowing for larger number of rounds for experiments
            nn.ReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=1024,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=10),
        )

    def forward(self,x):
        x=self.conv_layer(x)
        x=x.view(-1,1024)
        logits=self.fc_layer(x)
        return F.log_softmax(logits,dim=1)     

# WY's 2nd cnn model based on TF tutorial, batch normalization is used
class CNNCifarTfBn(nn.Module):
    def __init__(self):
        super(CNNCifarTfBn,self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3), # output size 30*30, i.e., (32, 30 ,30)
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2,stride=2), # output size 15*15, i.e., (32, 15 ,15)
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3), # output size 13*13, i.e., (64, 13 ,13)
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2,stride=2), # output size 6*6, i.e., (64, 6, 6)
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3), # output size 4*4, i.e., (64, 4, 4)
            # nn.Dropout2d(), # this is WY's added layer to postpone overfitting, allowing for larger number of rounds for experiments
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=1024,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=10),
        )

    def forward(self,x):
        x=self.conv_layer(x)
        x=x.view(-1,1024)
        logits=self.fc_layer(x)
        return F.log_softmax(logits,dim=1) 


# the 2-NN model described in the vanilla FL paper for experiments with MNIST
class TwoNN(nn.Module):
    def __init__(self):
        super(TwoNN,self).__init__()
        self.nn_layer=nn.Sequential(
            nn.Linear(in_features=28*28,out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100,out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100,out_features=10)
        )
    def forward(self,x):
        x = x.view(-1,28*28)
        logits = self.nn_layer(x)
        return F.log_softmax(logits,dim=1)
                 
        
# the CNN model described in the vanilla FL paper for experiments with MNIST
class CNNMnistWy(nn.Module):
    def __init__(self):
        super(CNNMnistWy,self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=1024,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=10),
        )
    
    def forward(self,x):
        x=self.conv_layer(x)
        x=x.view(-1,1024)
        logits = self.fc_layer(x)
        return F.log_softmax(logits,dim=1)
        
def get_count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)