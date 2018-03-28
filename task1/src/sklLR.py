from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
import pandas as pd
import numpy as np
import pickle 

np.random.seed(1)
# hyper parameters
alpha_val = 0.10
learning_rate = 0.01
training_rounds = 200


with open('../data/training_vec.pk1', 'rb') as f:
	X_train = pickle.load(f)
	y_train = pickle.load(f)
	print (X_train)
y_train = np.array(y_train)
#iris = 
print (X_train.shape)
#y_train = y_train[0:len(X_train)]

#print (X_train)
print (X_train.shape)
#print (y_train)
print (y_train.shape)
feature_num = X_train[0].shape[1]

# partition
train_indices = np.random.choice(X_train.shape[0], round(X_train.shape[0]*0.8), replace = False)
test_indices = np.array(list(set(range(X_train.shape[0])) - set(train_indices)))
X_train_train = X_train[train_indices]
X_train_test = X_train[test_indices]
y_train_train = y_train[train_indices]
y_train_test = y_train[test_indices]

# model and loss function's definition
batch_size = 100
classifier = LogisticRegression(penalty='l1', class_weight='balanced')
classifier.fit(X_train, y_train)
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print ('accuracy:', np.mean(scores), scores)
print (len(scores))

with open('../data/test_vec.pk1', 'rb') as f:
	X_test = pickle.load(f)

ans = classifier.predict_proba(X_test)
with open('../data/ans.csv', 'w') as f:
	for i in range(len(ans)):
		f.write(str(ans[i][1]))
		f.write('\n')
