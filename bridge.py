#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 19:50:08 2017

@author: yishao
"""

import pickle
import pandas as pd
import os
import numpy as np   

from sklearn.svm import LinearSVC    
with open('svm_models/svm_0.5.p', 'rb') as f:
    trained_model = pickle.load(f)
    
 
with open('processed_data/data_tfidf.p', 'rb') as f:
    train_p = pickle.load(f).toarray()
    train_dev_p = pickle.load(f).toarray()
    dev_p = pickle.load(f).toarray()
    test_p = pickle.load(f).toarray()
    
train = pd.read_csv('data/train.csv')
train_dev = pd.read_csv('data/train_dev.csv')
dev = pd.read_csv('data/dev.csv')
test = pd.read_csv('data/test.csv')

from sklearn import preprocessing
classes = train['class'].unique()
le = preprocessing.LabelEncoder()
le.fit(classes)

train_classes = le.transform(train['class'])
train_dev_classes=le.transform(train_dev['class'])



    
    
predicted_class=trained_model.predict(dev.p)
for i in ptredicted_class:
    