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

def predict_token_nb(freq_map, data):
    data['predicted'] = data.apply(lambda r: find_freq_token(r['before'], freq_map), axis = 1)

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

# Prepare the classes
train = pd.read_csv('data/train.csv')
classes = train['class'].unique()
encoder = preprocessing.LabelEncoder()
encoder.fit(classes)

# Load the testing data
test = pd.read_csv('data/test.csv')

# Prepare the testing data
test_before = test['before'].tolist()
test_after = test['after'].tolist()
test_class = test['class'].tolist()

# SVM model
svm_test = pp.pipeline_predict(token_nb_model, svm_model, nh.normalize_token, 
                               test_before, train_tfidf, encoder, 
                               test_after, test_class)
svm_test_acc = pp.find_accuracy(svm_test)

# Multinomial Naive Bayes model
mnb_test = pp.pipeline_predict(token_nb_model, char_nb_model, nh.normalize_token, 
                               test_before, train_tfidf, encoder, 
                               test_after, test_class)
mnb_test_acc = pp.find_accuracy(mnb_test)



# Log the process
with open('final_analysis/test_accuracy.txt', 'w') as f:
    f.write('SVM model' + '\n')
    f.write('testing accuracy: ' + str(svm_test_acc) + '\n')



