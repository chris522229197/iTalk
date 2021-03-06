import pandas as pd
import time
import pickle

train_data = pd.read_csv('data/train.csv', index_col = 0)
train_dev_data = pd.read_csv('data/train_dev.csv', index_col = 0)

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
def log_token_nb(train_time, train_predict_time, train_accuracy, 
                 dev_predict_time, dev_accuracy):
    with open('naive_bayes/token_nb_log.txt', 'w') as log_file:
        log_file.write('train time: ' + str(int(train_time)) + '\n')
        log_file.write('train prediction time: ' + str(int(train_predict_time)) + '\n')
        log_file.write('train accuracy: ' + str(train_accuracy) + '\n')
        log_file.write('dev prediction time: ' + str(int(dev_predict_time)) + '\n')
        log_file.write('dev accuracy: ' + str(dev_accuracy) + '\n')
    
# Train on the train data
print('Training on the train data ...')
train0 = time.clock()
token_nb_model = train_token_nb(train_data)
with open('naive_bayes/token_nb_model.p', 'wb') as f:
    pickle.dump(token_nb_model, f)
train1 = time.clock()

# Compute the training accuracy
print('Finding training accuracy ...')
train_predict0 = time.clock()
predict_token_nb(token_nb_model, train_data)
train_accuracy = sum(train_data['after'] == train_data['predicted']) / train_data.shape[0]
train_predict1 = time.clock()

# Compute the development accuracy and output result
print('Finding development accuracy ...')
dev_predict0 = time.clock()
predict_token_nb(token_nb_model, train_dev_data)
dev_accuracy = sum(train_dev_data['after'] == train_dev_data['predicted']) / train_dev_data.shape[0]
with open('naive_bayes/dev_prediction.p', 'wb') as f:
    pickle.dump(train_dev_data, f)
dev_predict1 = time.clock()

# Log the process
log_token_nb(train1 - train0, train_predict1 - train_predict0, train_accuracy, 
             dev_predict1 - dev_predict0, dev_accuracy)
print('Done!')
