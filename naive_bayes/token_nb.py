import pandas as pd
import time
import pickle

train_data = pd.read_csv('data/shuffled_train.csv', index_col = 0)
dev_data = pd.read_csv('data/shuffled_dev.csv', index_col = 0)

# For a given before token, find the frequencies of its corresponding after tokens
def count_token(before, after, freq_map):
    if before not in freq_map:
        freq_map[before] = {after: 1}
    elif after not in freq_map[before]:
        freq_map[before][after] = 1
    else:
        freq_map[before][after] += 1
    
# Train a Naive Bayes model for each unique before token
def train_token_nb(train):
    freq_map = {}
    train.apply(lambda r: count_token(r['before'], r['after'], freq_map), axis = 1)
    return freq_map


# Find the after token with the highest frequency
def find_freq_token(before, freq_map):
    if before not in freq_map:
        return before
    else:
        return max(freq_map[before], key = freq_map[before].get)

# Predict the after token based on the highest frequency
def predict_token_nb(freq_map, dev):
    dev['predicted'] = dev.apply(lambda r: find_freq_token(r['before'], freq_map), axis = 1)

# Log the process
def log_token_nb(train_time, predict_time, accuracy):
    with open('naive_bayes/token_nb_log.txt', 'w') as log_file:
        log_file.write('training time: ' + str(int(train_time)) + '\n')
        log_file.write('prediction time: ' + str(int(predict_time)) + '\n')
        log_file.write('dev accuracy:' + str(accuracy) + '\n')
    
# Train
train0 = time.clock()
token_nb_model = train_token_nb(train_data)
with open('naive_bayes/token_nb_model.p', 'wb') as model_file:
    pickle.dump(token_nb_model, model_file)
train1 = time.clock()

# Predict
predict0 = time.clock()
predict_token_nb(token_nb_model, dev_data)

with open('naive_bayes/token_nb_prediction.p', 'wb') as predict_file:
    pickle.dump(dev_data, predict_file)
predict1 = time.clock()

# Find accuracy
accuracy = sum(dev_data['after'] == dev_data['predicted']) / dev_data.shape[0]

# Log the process
log_token_nb(train1 - train0, predict1 - predict0, accuracy)
