import pandas as pd
dev_data = pd.read_csv('data/shuffled_dev.csv', index_col = 0)
accuracy = sum(dev_data['before'] == dev_data['after']) / dev_data.shape[0]

with open('benchmark.txt', 'w') as log_file:
    log_file.write('Benchmark: predict the after token as the before token' + '\n')
    log_file.write('dev accuracy: ' + str(accuracy) + '\n')

