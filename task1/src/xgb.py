import xgboost as xgb
from scipy.sparse import coo_matrix, hstack
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import pandas as pd
import numpy as np
import pickle 
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,4

np.random.seed(1)
# hyper parameters
#  alpha_val = 0.10
#  learning_rate = 0.01
#  training_rounds = 200

with open('../data/training_vec.pk1', 'rb') as f:
	X_train = pickle.load(f)
	y_train = pickle.load(f)
#	y_train = coo_matrix(np.array([y_train]).T.astype(int))
	y_train = np.array(y_train).astype(int)
	y_train = np.array([(val == 1) for val in y_train])
#	print (X_train.shape)
#	print (y_train.shape)
#	print (X_train)
#	print (y_train)
#	train = hstack([X_train, y_train])
#	print (train.shape)
with open('../data/test_vec.pk1', 'rb') as f:
	X_test = pickle.load(f)

feature_num = X_train.shape[1]

# target = 'Disbursed'
# IDcol = 'ID'

def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=5000):
	if useTrainCV:
		xgb_param = alg.get_xgb_params()
		xgtrain = xgb.DMatrix(X_train, label = y_train)
		cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
				metrics='auc', early_stopping_rounds=early_stopping_rounds)
		alg.set_params(n_estimators=cvresult.shape[0])

	# Fit the algorithm on the data
	bst = alg.fit(X_train, y_train, eval_metric='auc')

	# Predict training set:
	dtrain_predictions = alg.predict(X_train)
	dtrain_predprob = alg.predict_proba(X_train)[:,1]

	# Print model report:
	print ('\nModel Report')
	print ('Accuracy : %.4g' % metrics.accuracy_score(y_train, dtrain_predictions))
	print ('AUC Score (Train): %f' % metrics.roc_auc_score(y_train, dtrain_predprob))

#	feat_imp = pd.Series(alg.feature_importances_).sort_values(ascending=False)
#	feat_imp.plot(kind='bar', title='Feature Importances')
#	plt.ylabel('Feature Importance Score')


xgb1 = XGBClassifier(
	learning_rate = 0.1,
#	updater = 'grow_gpu',
	n_estimators = 1000,
	max_depth = 12,
	min_child_weight = 1,
	gamma = 0,
	subsample = 0.8,
	colsample_bytree = 0.8,
	objective = 'binary:logistic',
	n_jobs = 4,
	scale_pos_weight = 1,
	seed = 27)
modelfit(xgb1, X_train, y_train)

y_test = xgb1.predict(X_test)
y_prob = xgb1.predict_proba(X_test)[:,1]
print (y_test)
print (y_prob)
with open('../data/ans.csv', 'w') as f:
	for i in range(len(y_prob)):
		f.write(str(y_prob[i]))
		f.write('\n')
'''
# partition
train_indices = np.random.choice(X_train.shape[0], round(X_train.shape[0]*0.8), replace = False)
test_indices = np.array(list(set(range(X_train.shape[0])) - set(train_indices)))
train_X = X_train[train_indices]
test_X = X_train[test_indices]
train_Y = y_train[train_indices]
test_Y = y_train[test_indices]

# model and loss function's definition
xgm = XGBClassifier()
xgm.fit(X_train, y_train)
y_pred = xgm.predict(X_train)
predictions = [round(value) for value in y_pred]
predprob = xgm.predict_proba(X_train)[:,1]
accuracy = metrics.accuracy_score(y_train, predictions)
print ("XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)

param = {}
param['gpu_id'] = 1
param['updater'] = 'gpu_hist'
param['seed'] = 0
param['objective'] = 'binary:logistic'
param['eta'] = 0.1
param['max_depth'] = 3
param['silent'] = True
param['nthread'] = -1
param['num_class'] = 2

watchlist = [ (xg_train, 'train'), (xg_test, 'test') ]
num_round = 100
print ('training')
bst = xgb.train(params=param, dtrain=xg_train, num_boost_round=num_round, evals=watchlist)

print ('predicting')
pred = bst.predict(xg_test)

print ('predicting, classification error=%f' % (sum(int(pred[i])!=test_Y[i] in range(len(test_Y))) / float(len(test_Y)) ))


'''
