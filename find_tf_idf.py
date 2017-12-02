import pandas as pd
import text_helpers as th
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import time

t0 = time.clock()

all_data = pd.read_csv('data/en_train.csv')
train_data = pd.read_csv('data/train.csv', index_col=0)
train_dev_data = pd.read_csv('data/train_dev.csv', index_col=0)
dev_data = pd.read_csv('data/dev.csv', index_col=0)
test_data = pd.read_csv('data/test.csv', index_col=0)

# Exclude tokens that are non-English characters (e.g Asian and Arabic characters)
non_verbatim = all_data[all_data['class'] != 'VERBATIM']
verbatim_english = all_data[(all_data['class'] == 'VERBATIM') & 
                            (all_data['before'] != all_data['after'])]
english_data = non_verbatim.append(verbatim_english)

# Find the set of unique characters (vocabulary)
vocab = th.find_vocab(english_data, 'before')

# Create a index reference for the unique characters
idx_lookup, char_lookup = th.create_idx(vocab)

# Find the term frequency matrix for the split data sets
train_tf = th.count_char(train_data, 'before', idx_lookup)
train_dev_tf = th.count_char(train_dev_data, 'before', idx_lookup)
dev_tf = th.count_char(dev_data, 'before', idx_lookup)
test_tf = th.count_char(test_data, 'before', idx_lookup)

# Find the normalized smooth idf based on the training data
train_idf = TfidfTransformer(norm='l2')
train_idf.fit(vstack([train_tf, train_dev_tf]))

# Find the tf-idf matrix for each data subset
train_tfidf = train_idf.transform(train_tf)
train_dev_tfidf = train_idf.transform(train_dev_tf)
dev_tfidf = train_idf.transform(dev_tf)
test_tfidf = train_idf.transform(test_tf)

# Save the data
with open('processed_data/data_tfidf.p', 'wb') as tfidf_file:
    pickle.dump(train_tfidf, tfidf_file)
    pickle.dump(train_dev_tfidf, tfidf_file)
    pickle.dump(dev_tfidf, tfidf_file)
    pickle.dump(test_tfidf, tfidf_file)

with open('processed_data/data_tf.p', 'wb') as tf_file:
    pickle.dump(train_tf, tf_file)
    pickle.dump(train_dev_tf, tf_file)
    pickle.dump(dev_tf, tf_file)
    pickle.dump(test_tf, tf_file)

with open('processed_data/char_idx_lookup.p', 'wb') as lookup_file:
    pickle.dump(idx_lookup, lookup_file)
    pickle.dump(char_lookup, lookup_file)

t1 = time.clock()

# Log the process
with open('processed_data/tfidf_log.txt', 'w') as log_file:
    log_file.write('processing time: ' + str(int(t1 - t0)) + '\n')
    log_file.write('train shape: ' + str(train_tfidf.shape) + '\n')
    log_file.write('train dev shape: ' + str(train_dev_tfidf.shape) + '\n')
    log_file.write('dev shape: ' + str(dev_tfidf.shape) + '\n')
    log_file.write('test shape: ' + str(test_tfidf.shape) + '\n')
    