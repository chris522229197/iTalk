#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:39:23 2017

@author: Sarah
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
#from sklearn import svm, datasets
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#from sklearn import preprocessing

## import some data to play with
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target
#class_names = iris.target_names
#
## Split the data into a training set and a test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
## Run classifier, using a model that is too regularized (C too low) to see
## the impact on the results
#classifier = svm.SVC(kernel='linear', C=0.01)
#y_pred = classifier.fit(X_train, y_train).predict(X_test)

## Encode classes
#train_dev_pd= pd.read_csv('data/train_dev.csv')
#class_encoder = pickle.load(open('classEncoder.p', 'rb'))
#train_dev_class = class_encoder.transform(train_dev_pd['class'])

# true train_dev class
train_dev_pd= pd.read_csv('data/train_dev.csv')
class_names = train_dev_pd['class'].unique()
y_true = train_dev_pd['class']

# predict train_dev class with SVM 
with open('processed_data/data_tfidf.p', 'rb') as f:
    train = pickle.load(f).toarray()
    train_dev = pickle.load(f).toarray()
svm_model = pickle.load(open('svm_model.p', 'rb'))
prediction = svm_model.predict(train_dev)
# transform back to class
class_encoder = pickle.load(open('classEncoder.p', 'rb'))
classes = class_encoder.inverse_transform(range(16))
y_pred = class_encoder.inverse_transform(prediction)

def plot_confusion_matrix(y_true, y_pred, classes,
                          cmap=plt.cm.Reds):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)
    cm[cm < 0.01] = 0
    print("Normalized confusion matrix")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')

#np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
plt.figure(figsize=(100, 100))
plot_confusion_matrix(y_true, y_pred, classes)
plt.show()
plt.savefig('Confusion Matrix',bbox_inches='tight')

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# unbalanced
# evaluation for each class

cm = confusion_matrix(y_true, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
unb_accuracys = np.round(np.diag(cm), 2)

unb_precisions = np.round(precision_score(y_true, y_pred, average= None), 2)
unb_recalls = np.round(recall_score(y_true, y_pred, average= None), 2)
df_un = pd.DataFrame({'Class': classes,'Accuracy':unb_accuracys,
                      'Precision':unb_precisions,
                      'Recall':unb_recalls})
df_un = df_un[['Class','Accuracy', 'Precision', 'Recall']]
writer = pd.ExcelWriter('SVM_Metrics_Class.xlsx', engine='xlsxwriter')
df_un.to_excel(writer, sheet_name='Sheet1')
worksheet = writer.sheets['Sheet1']
worksheet.conditional_format('C2:B17', {'type': '3_color_scale'})
writer.save()



# average values
unbalanced_accuracy = 98.54
unbalanced_precision =  np.round(precision_score(y_true, y_pred, average= 'weighted')*100,2)
unbalanced_recall =  np.round(recall_score(y_true, y_pred, average= 'weighted')*100,2)

# balanced svm
svm_model_ba = pickle.load(open('svm_balanced_2.p', 'rb'))
prediction_ba = svm_model_ba.predict(train_dev)
y_pred_ba = class_encoder.inverse_transform(prediction_ba)
balanced_accuracy = 98.01
balanced_precision =  np.round(precision_score(y_true, y_pred_ba, average= 'weighted')*100,2)
balanced_recall =  np.round(recall_score(y_true, y_pred_ba, average= 'weighted')*100,2)

df = pd.DataFrame({'Parameter': ['unbalanced','balanced'],
                   'Accuracy':[unbalanced_accuracy,balanced_accuracy],
                   'Precision':[unbalanced_precision,balanced_precision],
                   'Recall':[unbalanced_recall,balanced_recall]})
df = df[['Parameter','Accuracy', 'Precision', 'Recall']]
writer = pd.ExcelWriter('SVM_Compare.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
#worksheet = writer.sheets['Sheet1']
#worksheet.conditional_format('C2', {'type': '3_color_scale'})
writer.save()
