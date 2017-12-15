#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:03:58 2017

@author: Sarah
"""
from sklearn import preprocessing
import pickle
import pandas as pd
train_pd= pd.read_csv('data/train.csv')
classes = train_pd['class'].unique()
le = preprocessing.LabelEncoder()
le.fit(classes)
# save to file
le_filename = 'classEncoder.p'
le_model_pkl = open(le_filename, 'wb')
pickle.dump(le, le_model_pkl)
le_model_pkl.close()