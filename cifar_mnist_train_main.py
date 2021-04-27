# This script trains some simple CNN and NN with CIFAR10 and MNIST datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
from torchvision import transforms
from torchvision import datasets

from cifar_mnist_dataloader import data_cifar, data_mnist
from my_nn_models import CNNCifarTorch, CNNCifarTf, TwoNN, CNNMnistWy, get_count_params

class TaskMnist():
    def __init__(self, nn='cnn_wy'):
        self.path = './data/mnist'
        self.name = 'mnist'
        self.nn = nn
        
class TaskCifar():
    def __init__(self,nn='cnn_torch'):
        self.path = './data/cifar'
        self.name = 'cifar'
        self.nn = nn

class HyperParam():
    def __init__(self,path,learning_rate=0.1, batch_size=100, epoch=10, momentum=0.9, nesterov=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.datapath = path
        self.lr=learning_rate
        self.bs=batch_size
        self.epoch=epoch
        self.momentum=momentum
        self.nesterov=nesterov        

# training function
def train_model(loader_train, loader_test, epochs, loss_fn, optimizer, device):
    for epoch in range(1, epochs+1):
        train_loss = 0.0
        test_loss = 0.0
        test_acc = 0.0

        # training of each epoch
        model.train()
        for batch, (images, labels) in enumerate(loader_train):
            images, labels = images.to(device), labels.to(device)
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
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs,labels)
                test_loss += loss.item() * images.size(0)
                pred = outputs.argmax(dim=1)
                num_correct += pred.eq(labels.view_as(pred)).sum().item()
        test_loss /= len(loader_test.dataset)
        test_acc = 100*num_correct/len(loader_test.dataset)
        print('Epoch: {:>4}/{} | Training Loss: {:.2f} | Test Loss: {:.2f} | Test accuracy = {:.2f}%'.format(epoch, epochs, train_loss, test_loss, test_acc))

if __name__ == '__main__':
    torch.manual_seed(1)
    # configure the task and training settings
    # task = TaskMnist(nn='2nn_wy')    
    task = TaskCifar(nn='cnn_torch')
    settings = HyperParam(path=task.path, learning_rate=0.1, epoch=10, nesterov=False)  
    
    if task.name == 'mnist':
        if task.nn == 'cnn_wy':
            model = CNNMnistWy().to(settings.device)
        elif task.nn == '2nn_wy':
            model = TwoNN().to(settings.device)
        loader_train, loader_test = data_mnist(path=settings.datapath,batch_size=settings.bs)
    elif task.name == 'cifar':
        if task.nn == 'cnn_torch':
            model = CNNCifarTorch().to(settings.device)
        else:
            model = CNNCifarTf().to(settings.device)
        loader_train, loader_test = data_cifar(path=settings.datapath,batch_size=settings.bs)
    
    # set the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss().to(settings.device)
    if settings.nesterov:
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.lr, momentum=settings.momentum, nesterov=settings.nesterov)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.lr)
    
    # print some welcome messages
    print('\nModel training initiated...\n')
    print(f'Dataset:\t{task.name}')
    print(f'Loss function:\t{loss_fn}')
    print('Optimizer:\tSGD with Nesterov momentum=0.9') if settings.nesterov else print('Optimizer:\tvanilla SGD')
    print(f'learning rate:\t{settings.lr}')
    print(f'Batch size:\t{settings.bs}')
    print(f'Num of epochs:\t{settings.epoch}')
    print('Model to train:\n', model)
    print(f'Trainable model parameters:\t{get_count_params(model)}')

    # start training
    start = time.time()
    train_model(loader_train=loader_train,
                loader_test=loader_test,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=settings.epoch,
                device=settings.device)

    # print the wall-clock-time used
    end=time.time() 
    print('\nWall clock time elapsed: {:.2f}s'.format(end-start))
