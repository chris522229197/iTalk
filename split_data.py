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
    log_file = open('data/split_log.txt', 'w')
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
DEV_SIZE = 150000
SEED = 123

# Set the seed for random generation
rd.seed(SEED)

# Import data
raw_data = pd.read_csv(RAW_DATA_PATH)
num_sentences = len(raw_data['sentence_id'].unique())

# Split the data based on the raw data ordering
ordered_train_ids = list(range(TRAIN_SIZE))
ordered_dev_ids = list(range(TRAIN_SIZE, TRAIN_SIZE + DEV_SIZE))
ordered_test_ids = list(range(TRAIN_SIZE + DEV_SIZE, num_sentences))

ordered_train = subset_sentence(raw_data, ordered_train_ids)
ordered_dev = subset_sentence(raw_data, ordered_dev_ids)
ordered_test = subset_sentence(raw_data, ordered_test_ids)

# Split the data based on shuffled ordering
shuffled_ids = rd.sample(list(range(num_sentences)), num_sentences)

shuffled_train_ids = shuffled_ids[:TRAIN_SIZE]
shuffled_dev_ids = shuffled_ids[TRAIN_SIZE:(TRAIN_SIZE + DEV_SIZE)]
shuffled_test_ids = shuffled_ids[(TRAIN_SIZE + DEV_SIZE):num_sentences]

shuffled_train = subset_sentence(raw_data, shuffled_train_ids)
shuffled_dev = subset_sentence(raw_data, shuffled_dev_ids)
shuffled_test = subset_sentence(raw_data, shuffled_test_ids)

# Save the split data as csv files
split_data = dict(zip(['ordered_train', 'ordered_dev', 'ordered_test', 
                       'shuffled_train', 'shuffled_dev', 'shuffled_test'], 
                      [ordered_train, ordered_dev, ordered_test, 
                       shuffled_train, shuffled_dev, shuffled_test]))
batch_save(split_data)
split_log(SEED, split_data)



