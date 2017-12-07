#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import numpy as np   
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import normalization_helpers as nh

# Load the best validated SVM model
with open('svm_models/svm_0.5.p', 'rb') as f:
    svm_model = pickle.load(f)
    
# Load the token-level Naive Bayes model
with open('naive_bayes/token_nb_model.p', 'rb') as f:
    token_nb_model = pickle.load(f)
    
with open('processed_data/data_tfidf.p', 'rb') as f:
    train_p = pickle.load(f).toarray()
    train_dev_p = pickle.load(f).toarray()
    dev_p = pickle.load(f).toarray()
    test_p = pickle.load(f).toarray()
    
train = pd.read_csv('data/train.csv')
train_dev = pd.read_csv('data/train_dev.csv')
dev = pd.read_csv('data/dev.csv')
test = pd.read_csv('data/test.csv')

classes = train['class'].unique()
le = preprocessing.LabelEncoder()
le.fit(classes)

# Prepare data
encoder = le
translator = nh.normalize_token

train_before = train['before'].tolist()
train_after = train['after'].tolist()

train_dev_before = train_dev['before'].tolist()
train_dev_after = train_dev['after'].tolist()

# Combine the token-level Naive Bayes, step 1 classification model, and step 2 translation
def pipeline_predict(token_nb, step1_model, step2_trans, tokens, tokens_tfidf, class_encoder):
    pred_tokens = []
    for i in range(len(tokens)):
        token = tokens[i]
        if token in token_nb:
            pred_token = max(token_nb[token], key = token_nb[token].get)
        else:
            token_tfidf = tokens_tfidf[i, :]
            token_tfidf = token_tfidf.reshape((1, token_tfidf.shape[0]))
            pred_class = step1_model.predict(token_tfidf)
            pred_class = class_encoder.inverse_transform(pred_class)[0]
            pred_token = step2_trans(token, pred_class)
            pred_token = token
        pred_tokens.append(pred_token)
    return pred_tokens
        
def find_matched_count(truth_list, pred_list):
    matched_count = 0
    for i in range(len(truth_list)):
        if truth_list[i] == pred_list[i]:
            matched_count += 1
    return matched_count

train_pred = pipeline_predict(token_nb_model, svm_model, translator, train_before, 
                              train_p, encoder)
train_accuracy = find_matched_count(train_after, train_pred) / len(train_after)

dev_pred = pipeline_predict(token_nb_model, svm_model, translator, train_dev_before, 
                            train_dev_p, encoder)
dev_accuracy = find_matched_count(train_dev_after, dev_pred) / len(train_dev_after)


    