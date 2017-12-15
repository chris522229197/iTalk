
# Step one: Supervised Learning classify "class"
import pickle
import os
import numpy as np    
import pandas as pd
import matplotlib as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import time

os.chdir('/Users/Sarah/Desktop/CS229/iTalk')

#with open('processed_data/char_idx_lookup.p', 'rb') as f:
#    idx_lookup = pickle.load(f)
#    char_lookup = pickle.load(f)

with open('processed_data/data_tfidf.p', 'rb') as f:
    train = pickle.load(f).toarray()
    train_dev = pickle.load(f).toarray()
    dev = pickle.load(f).toarray()
    test = pickle.load(f).toarray()
train_pd= pd.read_csv('data/train.csv')
# get true labels
classes = train_pd['class'].unique()
le = preprocessing.LabelEncoder()
le.fit(classes)


train_class = le.transform(train_pd['class'])

train_dev_pd = pd.read_csv('data/train_dev.csv')
train_dev_class = le.transform(train_dev_pd['class'])

dev_pd = pd.read_csv('data/dev.csv')
dev_class = le.transform(dev_pd['class'])

test_pd = pd.read_csv('data/test.csv')
test_class = le.transform(test_pd['class'])

# logistic regression
start = time.clock()
logistic_model = LogisticRegression(multi_class="multinomial",solver = "lbfgs")
logistic_model.fit(train,train_class)
#save model
logistic_pkl_filename = 'logistic_classifier.pkl'
logistic_model_pkl = open(logistic_pkl_filename, 'wb')
pickle.dump(logistic_model, logistic_model_pkl)
logistic_model_pkl.close()
# predict on train_dev
logistic_prediction = logistic_model.predict(train_dev)
logistic_true_prediction = np.sum(logistic_prediction == train_dev_class)
logistic_accuracy = logistic_true_prediction/len(train_dev_class)
end = time.clock()
print('Logistic_model takes %d second to run',str(end-start))

# K-nearest neighbors
start = time.clock()
neigh_model = KNeighborsClassifier(n_neighbors = 5)
neigh_model.fit(train,train_class)
#save model
neigh_pkl_filename = 'neigh_classifier.pkl'
neigh_model_pkl = open(neigh_pkl_filename, 'wb')
pickle.dump(neigh_model, neigh_model_pkl)
neigh_model_pkl.close()
# predict on train_dev
neigh_prediction = neigh_model.predict(train_dev)
neigh_true_prediction = np.sum(neigh_prediction == train_dev_class)
neigh_accuracy = neigh_true_prediction/len(train_dev_class)
end = time.clock()
print('Logistic_model takes %d second to run',str(end-start))

#
#start = time.clock()
#neigh_model = KNeighborsClassifier(n_neighbors = 3)
#neigh_model.fit(train[0:100],train_class[0:100])
#neigh_prediction = neigh_model.predict(train_dev[0:100])
#neigh_true_prediction = np.sum(neigh_prediction[0:100] == train_class[0:100])
#neigh_accuracy = neigh_true_prediction/len(train_class[0:100])
#end = time.clock()
#print('Logistic_model takes %d second to run',str(end-start))


# predict model
best_model = svm??
# load naive bayes model
with open('token_nb_prediction.p','rb') as f:
    naive_bayes_model = pickle.load(f)

prediction = []    
for i in range(0,len(dev_pd['before'])):
    token = train_dev_pd['before'][i]
    token_tfidf = train_dev[i]
    if token in dict:
        # use naive bayes
        prediction.append(predict_token_nb(naive_bayes_model, token))
    else:
        # use svm with tf-idf
        prediction.append(svm_model.predict(token_tfidf))
true_prediction = np.sum(prediction == dev_class)
neigh_accuracy = neigh_true_prediction/len(devclass)