#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

# Load data
all_data = pd.read_csv('data/en_train.csv')
train_data = pd.read_csv('data/train.csv')
train_dev_data = pd.read_csv('data/train_dev.csv')

# Load the tf-idf matrices as arrays
with open('processed_data/data_tfidf.p', 'rb') as f:
    train = pickle.load(f).toarray()
    train_dev = pickle.load(f).toarray()
    dev = pickle.load(f).toarray()
    test = pickle.load(f).toarray()
    
# Find the token classes
classes = all_data['class'].unique()
encoded_classes = preprocessing.LabelEncoder()
encoded_classes.fit(classes)

# Encode the token classes
train_class = encoded_classes.transform(train_data['class'])
train_dev_class = encoded_classes.transform(train_dev_data['class'])

# Train the logistic regression model
X = train[0:100]
y = train_class[0:100]

test = train[101:200]
test_class = train_class[101:200]


logistic = LogisticRegression(multi_class='ovr', solver='liblinear', verbose=1)
logistic.fit(train, train_class)

with open('step1_logistics/models/model1.p', 'wb') as f:
    pickle.dump(logistic, f)

with open('step1_logistics/models/model1.p', 'rb') as f:
    loaded = pickle.load(f)



pred = logistic.predict(test)

accuracy = np.sum(pred == test_class) / len(test_class)