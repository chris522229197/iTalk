#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 19:39:28 2017

@author: yishao
"""

import pickle
import pandas as pd
import os

os.chdir('D:\Software\Dropbox\2. Academic\iTalk')

with open('processed_data/char_idx_lookup.p', 'rb') as f:
    idx_lookup = pickle.load(f)
    char_lookup = pickle.load(f)
import numpy as np    
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


from sklearn.svm import LinearSVC
clf = LinearSVC(C=0.5,max_iter=10000)
clf.fit(train_p, train_classes)
train_dev_pred=clf.predict(train_dev_p)

tran_dev_accuracy=np.sum(train_dev_pred==train_dev_classes)/len(train_dev_pred)
