# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:46:43 2018

@author: Zongyue Qin
"""

import pickle
import numpy as np

file_loc = '../dataset/cifar-10-batches-py/'
train_data_prefix = '/data_batch_'
test_data_file = '/test_batch'
meta_file = '/batches.meta'

def load_data():
    X=None
    y=None
    Xtest=None
    ytest=None
    labels = None
    
    filename = file_loc + meta_file
    with open(filename, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
        labels = data.get(b'label_names')
    
    filename = file_loc+train_data_prefix+'1'
    with open(filename, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
        X = data.get(b'data')
        y = data.get(b'labels')
    
    for i in range(2,6):
        filename = file_loc + train_data_prefix+str(i)
        with open(filename, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
            Xtmp = data.get(b'data')
            ytmp = data.get(b'labels')
            X = np.concatenate((X, Xtmp))
            y = np.concatenate((y, ytmp))
    
        
    filename = file_loc+test_data_file
    with open(filename, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
        Xtest = data.get(b'data')
        ytest = data.get(b'labels')
        
    return X, y, Xtest, ytest, labels