#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import normalization_helpers as nh

# Combine the token-level Naive Bayes, step 1 classification model, and step 2 translation
# to output the prediction on transformed token
def pipeline_predict(token_nb, step1_model, step2_trans, tokens, tokens_tfidf, class_encoder, 
                     correct_tokens, correct_classes):
    pred_tokens = []
    pred_classes = []
    for i in range(len(tokens)):
        token = tokens[i]
        if token in token_nb:
            pred_token = max(token_nb[token], key = token_nb[token].get)
            pred_classes.append('NONE')
        else:
            token_tfidf = tokens_tfidf[i, :]
            token_tfidf = token_tfidf.reshape((1, token_tfidf.shape[0]))
            pred_class = step1_model.predict(token_tfidf)
            pred_class = class_encoder.inverse_transform(pred_class)[0]
            pred_classes.append(pred_class)
            pred_token = step2_trans(token, pred_class)
        pred_tokens.append(pred_token)
    predictions = pd.DataFrame({'before': tokens, 
                                'class': correct_classes, 
                                'after': correct_tokens, 
                                'predicted_class': pred_classes, 
                                'predicted_after': pred_tokens})
    return predictions