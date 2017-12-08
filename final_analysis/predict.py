#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import prediction_pipeline as pp
import normalization_helpers as nh
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# Load the token-level Naive Bayes models
with open('naive_bayes/token_nb_model.p', 'rb') as f:
    token_nb_model = pickle.load(f)

# Load the best validated SVM model
with open('svm_models/svm_0.5.p', 'rb') as f:
    svm_model = pickle.load(f)

# Load the multinomial Naive Bayes model
with open('step1_char_nb/step1_nb_model.p', 'rb') as f:
    char_nb_model = pickle.load(f)

# Load the tf-idf data
with open('processed_data/data_tfidf.p', 'rb') as f:
    train_tfidf = pickle.load(f).toarray()
    train_dev_tfidf = pickle.load(f).toarray()
    dev_tfidf = pickle.load(f).toarray()
    test_tfidf = pickle.load(f).toarray()

# Load the training and development data
train = pd.read_csv('data/train.csv')
train_dev = pd.read_csv('data/train_dev.csv')

classes = train['class'].unique()
encoder = preprocessing.LabelEncoder()
encoder.fit(classes)

# Prepare the data
train_before = train['before'].tolist()
train_after = train['after'].tolist()
train_class = train['class'].tolist()

train_dev_before = train_dev['before'].tolist()
train_dev_after = train_dev['after'].tolist()
train_dev_class = train_dev['class'].tolist()

# Predict for the SVM model
svm_train = pp.pipeline_predict(token_nb_model, svm_model, nh.normalize_token, 
                                train_before, train_tfidf, encoder, 
                                train_after, train_class)
svm_dev = pp.pipeline_predict(token_nb_model, svm_model, nh.normalize_token, 
                              train_dev_before, train_dev_tfidf, encoder, 
                              train_dev_after, train_dev_class)

# Predict for the multinomial Naive Bayes model
mnb_train = pp.pipeline_predict(token_nb_model, char_nb_model, nh.normalize_token, 
                                train_before, train_tfidf, encoder, 
                                train_after, train_class)
mnb_dev = pp.pipeline_predict(token_nb_model, char_nb_model, nh.normalize_token, 
                              train_dev_before, train_dev_tfidf, encoder, 
                              train_dev_after, train_dev_class)
# Save the results
with open('final_analysis/svm_train.p', 'wb') as f:
    pickle.dump(svm_train, f)
    
with open('final_analysis/svm_dev.p', 'wb') as f:
    pickle.dump(svm_dev, f)
    
with open('final_analysis/mnb_train.p', 'wb') as f:
    pickle.dump(mnb_train, f)
    
with open('final_analysis/mnb_dev.p', 'wb') as f:
    pickle.dump(mnb_dev, f)

# Compute accuracies
svm_train_acc = pp.find_accuracy(svm_train)
svm_dev_acc = pp.find_accuracy(svm_dev)
mnb_train_acc = pp.find_accuracy(mnb_train)
mnb_dev_acc = pp.find_accuracy(mnb_dev)

# Log the process
with open('final_analysis/accuracies.txt', 'w') as f:
    f.write('=' * 50 + '\n')
    f.write('SVM model' + '\n')
    f.write('training accuracy: ' + str(svm_train_acc) + '\n')
    f.write('development accuracy: ' + str(svm_dev_acc) + '\n')
    f.write('=' * 50 + '\n')
    f.write('Multinomial Naive Bayes model' + '\n')
    f.write('training accuracy: ' + str(mnb_train_acc) + '\n')
    f.write('development accuracy: ' + str(mnb_dev_acc) + '\n')
