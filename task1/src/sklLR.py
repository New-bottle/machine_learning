from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

np.random.seed(1)
# hyper parameters
alpha_val = 0.10
learning_rate = 0.01
training_rounds = 200


with open('../data/training_vec.pk1', 'rb') as f:
	X_train = pickle.load(f)
	y_train = pickle.load(f)
#iris = 
X_train = np.array(X_train)
y_train = np.array(y_train)
y_train = y_train[0:len(X_train)]

#print (X_train)
print (X_train.shape)
#print (y_train)
print (y_train.shape)
feature_num = X_train.shape[1]
print (X_train[0].shape)

# partition
train_indices = np.random.choice(len(X_train), round(len(X_train)*0.8), replace = False)
test_indices = np.array(list(set(range(len(X_train))) - set(train_indices)))
X_train_train = X_train[train_indices]
X_train_test = X_train[test_indices]
y_train_train = y_train[train_indices]
y_train_test = y_train[test_indices]

# model and loss function's definition
batch_size = 100
classifier = LogisticRegression()
classifier.fit(X_train_train, y_train_train)
scores = cross_val_score(classifier, X_train_test, y_train_test, cv=5)
print ('accuracy:', np.mean(scores), scores)
print (len(scores))

with open('../data/test_vec.pk1', 'rb') as f:
	X_test = pickle.load(f)

ans = classifier.predict_proba(X_test)
with open('../ans.csv', 'w') as f:
	for i in range(len(ans)):
		f.write(str(ans[i][0]))
		f.write('\n')
