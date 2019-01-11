# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:19:14 2018

@author: Zongyue Qin
"""
import utils
import numpy as np
import net
import torch

if __name__ == '__main__':
    
    # Load data
    X, y, Xtest, ytest, labels = utils.load_data()
    model = net.Model(epoch=50, batch_size=100,lr=3e-4)
    model.fit(X, y)

    # Compute test error
    ytest = torch.tensor(ytest, dtype=torch.int64)    
    t,_ = Xtest.shape
    error = 0

    # Because the memory limitation of my own computer, I cannot predict all at
    # once
    for i in range(0,t,50):
        yhat = model.predict(Xtest[i:i+50,:])
        error = error +\
        torch.sum(torch.tensor(yhat != ytest[i:i+50], dtype=torch.float)).item()

    print('test error = %f'%(error / t))


    # Compute training error
    y = torch.tensor(y, dtype=torch.int64)    
    n,_ = X.shape
    error = 0

    # Because the memory limitation of my own computer, I cannot predict all at
    # once
    for i in range(0,n,50):
        yhat = model.predict(X[i:i+50,:])
        error = error +\
        torch.sum(torch.tensor(yhat != y[i:i+50], dtype=torch.float)).item()

    print('training error = %f'%(error / n))
