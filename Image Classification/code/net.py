# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:31:52 2018

@author: Zongyue Qin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32*3*3, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, X):
        # transform X into a n*3*32*32 tensor
        X = X.view(-1, 3, 32, 32)
        X = F.max_pool2d(F.relu(self.conv1(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv2(X)), (2, 2), padding=1)
        X = F.max_pool2d(F.relu(self.conv3(X)), (2, 2), padding=1) 
        
        X = X.view(-1, self.num_features(X))
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X
        
    def num_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features = num_features * s
        return num_features
    
class Model:
    
    def __init__(self, batch_size=50, epoch=5, lr=1e-3):
        self.cnn = CNN()
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optim = torch.optim.SGD(self.cnn.parameters(), lr=self.lr, momentum=0.9,
                                     weight_decay = 0.1)
        self.batch_size = batch_size
        self.epoch = epoch

    def adjust_learning_rate(self, epoch):
        # Adjust learning rate automatically by epoch
        lr = self.lr * (0.1 ** (epoch // 15))
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
        
    def fit(self, X, labels):
        n, _ = X.shape

        X_valid = torch.tensor(X[-50:,:], dtype=torch.float)
        X = torch.tensor(X[:-50,:], dtype=torch.float)

    
        y_valid = torch.tensor(labels[-50:], dtype=torch.int64)
        y = torch.tensor(labels[:-50], dtype=torch.int64)
        n = n - 50
        
        
        epoch = 1
        while epoch <= self.epoch:
            print('Epoch %d...'%epoch)
            self.adjust_learning_rate(epoch)
            cnt = 0
            while cnt < n:
                batch = np.random.choice(n, size=self.batch_size, replace=False)
                cnt = cnt + self.batch_size
                self.cnn.zero_grad()
                output = self.cnn(X[batch,:])
                loss = self.criterion(output, y[batch])
                loss.backward()
                self.optim.step()


            self.cnn.zero_grad()
            output = self.cnn(X_valid)
            yhat = torch.argmax(output, dim=1)
            error = torch.mean(torch.tensor(yhat!=y_valid, dtype=torch.float)).item()
            print('After %d epoch(s), loss = %f, validation error = %f'%(epoch, loss, error))
            epoch = epoch + 1

        print('After fit')

            
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float)
        output = self.cnn(X)
        labels = torch.argmax(output, dim=1)
        return labels
        
        
        
