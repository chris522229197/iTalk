# Purpose: To split the original data in en_train.csv into training, development, and test
# sets.

import pandas as pd
import random as rd

# Helper functions
def subset_sentence(df, sentence_ids):
    return df[df['sentence_id'].isin(sentence_ids)]

def batch_save(dfs):
    for name, df in dfs.items():
        df.to_csv('data/' + name + '.csv')
    return

def split_log(seed, dfs):
    log_file = open('data/split_log2.txt', 'w')
    log_file.write('seed: ' + str(seed) + '\n')
    for name, df in dfs.items(): 
        log_file.write('=' * 15 + '\n')
        log_file.write(name + '\n')
        log_file.write('number of sentences: ' + str(len(df['sentence_id'].unique())) + '\n')
        log_file.write('number of tokens: ' + str(df.shape[0]) + '\n\n')
    log_file.close()
    return

# Global constant variables
RAW_DATA_PATH = 'data/en_train.csv'
TRAIN_SIZE = 500000
TRAIN_DEV_SIZE = 75000
DEV_SIZE = 75000
SEED = 123

# Set the seed for random generation
rd.seed(SEED)

# Import data
raw_data = pd.read_csv(RAW_DATA_PATH)
num_sentences = len(raw_data['sentence_id'].unique())

# Split the data based on shuffled ordering
shuffled_ids = rd.sample(list(range(num_sentences)), num_sentences)

train_ids = shuffled_ids[:TRAIN_SIZE]
train_dev_ids = shuffled_ids[TRAIN_SIZE:(TRAIN_SIZE + TRAIN_DEV_SIZE)]
dev_ids = shuffled_ids[(TRAIN_SIZE + TRAIN_DEV_SIZE):(TRAIN_SIZE + TRAIN_DEV_SIZE + DEV_SIZE)]
test_ids = shuffled_ids[(TRAIN_SIZE + TRAIN_DEV_SIZE + DEV_SIZE):num_sentences]

train = subset_sentence(raw_data, train_ids)
train_dev = subset_sentence(raw_data, train_dev_ids)
dev = subset_sentence(raw_data, dev_ids)
test = subset_sentence(raw_data, test_ids)

# Save the split data as csv files
split_data = dict(zip(['train', 'train_dev', 'dev', 'test'], [train, train_dev, dev, test]))
batch_save(split_data)
split_log(SEED, split_data)



