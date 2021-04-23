import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
from torchvision import transforms
from torchvision import datasets

from models import CNNCifar
from data_preparation import data_setup

class HyperParam():
    def __init__(self,path,learning_rate=0.1, batch_size=64, epoch=10, momentum=0.9, nesterov=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.datapath = path
        self.lr=learning_rate
        self.bs=batch_size
        self.epoch=epoch
        self.momentum=momentum
        self.nesterov=nesterov        

if __name__ == '__main__':
    settings = HyperParam(path='.\data\cifar')
    model = CNNCifar().to(settings.device)
    loader_train, loader_test = data_setup(path=settings.datapath,batch_size=settings.bs)
    loss_fn = nn.CrossEntropyLoss().to(settings.device)
    if settings.nesterov:
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.lr, momentum=settings.momentum, nesterov=settings.nesterov)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.lr)
    
    # print some welcome messages
    print('\nModel training initiated...\n')
    print('Dataset:\tCIFAR10')
    print(f'Loss function:\t{loss_fn}')
#     optimizer_selection = 'SGD with Nesterov momentum=0.9' if settings.nesterov else 'vanilla SGD'
#     print('Optimizer:\t',optimizer_selection)
    print('Optimizer:\tSGD with Nesterov momentum=0.9') if settings.nesterov else print('Optimizer:\tvanilla SGD')
    print(f'learning rate:\t{settings.lr}')
    print(f'Batch size:\t{settings.bs}')
    print(f'Num of epochs:\t{settings.epoch}')
    print('Model to train:\n', model)

    # start training
    for epoch in range(1, settings.epoch+1):
        train_loss = 0.0
        test_loss = 0.0
        test_acc = 0.0
        
        # training of each epoch
        model.train()
        for batch, (images, labels) in enumerate(loader_train):
            images, labels = images.to(settings.device), labels.to(settings.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(loader_train.dataset)

        # test after each epoch
        model.eval()
        num_correct = 0 
        with torch.no_grad():
            for batch, (images, labels) in enumerate(loader_test):
                images, labels = images.to(settings.device), labels.to(settings.device)
                outputs = model(images)
                loss = loss_fn(outputs,labels)
                test_loss += loss.item() * images.size(0)
                pred = outputs.argmax(dim=1)
                num_correct += pred.eq(labels.view_as(pred)).sum().item()
        test_loss /= len(loader_test.dataset)
        test_acc = 100*num_correct/len(loader_test.dataset)
        print('Epoch: {} | Training Loss: {:.2f} | Test Loss: {:.2f} | Test accuracy = {:.2f}%'.format(epoch, train_loss, test_loss, test_acc))
