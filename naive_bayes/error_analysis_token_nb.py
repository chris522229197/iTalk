import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open('naive_bayes/train_prediction.p', 'rb') as file:
    train_prediction = pickle.load(file)

wrong = train_prediction[train_prediction['after'] != train_prediction['predicted']]

# Examine whether the tokens are transformed
transform_ratio = sum(wrong['before'] != wrong['after']) / wrong.shape[0]
print('Ratio of actual transformation: ' + str(transform_ratio))

# Examine whether the predictions are different from the original
predict_transform_ratio = sum(wrong['before'] != wrong['predicted']) / wrong.shape[0]
print('Ratio of predicted transformation: ' + str(predict_transform_ratio))

# Examine the class distribution for the wrong prediction
class_counts = pd.value_counts(wrong['class'].values, sort=True)
class_counts.rename('wrong', inplace=True)

plt.figure(figsize = (12, 12))
class_counts.plot(kind='bar')
plt.title('Figure 1. Disribution of token class in incorrect predictions.')
plt.xlabel('Token class')
plt.ylabel('Frequency')
plt.savefig('naive_bayes/conut_per_class.png')

# Normalize the counts by the total counts in the whole train dev set
total_counts = pd.value_counts(train_prediction['class'].values)
total_counts.rename('train_dev', inplace=True)
merged = pd.concat([class_counts, total_counts], axis=1, join='inner')
merged['normalized'] = merged['wrong'] / merged['train_dev']

plt.figure(figsize = (12, 12))
merged.sort_values('normalized', ascending=False)['normalized'].plot(kind='bar')
plt.title('Figure 2. Normalized distribution of token class in incorrect predictions.')
plt.xlabel('Token class')
plt.ylabel('Normalized frequency')
plt.savefig('naive_bayes/normalized_conut_per_class.png')