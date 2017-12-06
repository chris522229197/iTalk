#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
import time

# Load data
all_data = pd.read_csv('data/en_train.csv')
train_data = pd.read_csv('data/train.csv')
train_dev_data = pd.read_csv('data/train_dev.csv')

# Load the tf sparse matrices
with open('processed_data/data_tf.p', 'rb') as f:
    train_tf = pickle.load(f)
    train_dev_tf = pickle.load(f)
    dev_tf = pickle.load(f)
    test_tf = pickle.load(f)
    
# Find the token classes
classes = all_data['class'].unique()
encoded_classes = preprocessing.LabelEncoder()
encoded_classes.fit(classes)

# Encode the token classes
train_class = encoded_classes.transform(train_data['class'])
train_dev_class = encoded_classes.transform(train_dev_data['class'])

# Prepare toy data sets
X = train_tf
y = train_class

test = train_dev_tf
test_class = train_dev_class

# Multinomial Naive Bayes
t0 = time.clock()
char_nb = MultinomialNB()
char_nb.fit(X, y)
char_nb.predict(test)
t1 = time.clock()

pred = char_nb.predict(test)
accuracy = np.sum(pred == test_class) / len(test_class)

