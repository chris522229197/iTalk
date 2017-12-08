#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
import time

# Log the process
def process_log(train_time, train_accuracy, dev_accuracy):
    with open('step1_char_nb/process_log.txt', 'w') as log_file:
        log_file.write('train time: ' + str(int(train_time)) + '\n')
        log_file.write('train accuracy (token class): ' + str(train_accuracy) + '\n')
        log_file.write('dev accuracy (token class): ' + str(dev_accuracy) + '\n')

# Load data
train_data = pd.read_csv('data/train.csv')
train_dev_data = pd.read_csv('data/train_dev.csv')

# Load the tf sparse matrices
with open('processed_data/data_tf.p', 'rb') as f:
    train_tf = pickle.load(f)
    train_dev_tf = pickle.load(f)
    dev_tf = pickle.load(f)
    test_tf = pickle.load(f)
    
# Find the token classes
classes = train_data['class'].unique()
encoded_classes = preprocessing.LabelEncoder()
encoded_classes.fit(classes)

# Encode the token classes
train_class = encoded_classes.transform(train_data['class'])
train_dev_class = encoded_classes.transform(train_dev_data['class'])

# Train the multinomial Naive Bayes with Laplace smoothing
train0 = time.clock()
char_nb_model = MultinomialNB()
char_nb_model.fit(train_tf, train_class)
with open('step1_char_nb/step1_nb_model.p', 'wb') as f:
    pickle.dump(char_nb_model, f)
train1 = time.clock()

# Compute the training accuracy
train_pred = char_nb_model.predict(train_tf)
train_accuracy = np.sum(train_pred == train_class) / len(train_class)

# Compute the development accuracy
dev_pred = char_nb_model.predict(train_dev_tf)
dev_accuracy = np.sum(dev_pred == train_dev_class) / len(train_dev_class)

# Log the process
process_log(train1 - train0, train_accuracy, dev_accuracy)

