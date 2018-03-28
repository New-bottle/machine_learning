import csv
import pickle


with open('../data/test.pk1', 'rb') as f:
	contents = pickle.load(f)
	titles = pickle.load(f)
	ids = pickle.load(f)

Y = []
with open('../data/ans.csv', 'r') as f:
	reader = f.readlines()
	for line in reader:
		Y.append(float(line))
print (len(ids))
print (len(Y))
with open('../data/submit.csv', 'w') as f:
	print ('id,pred', file = f)
	for i in range(len(ids)):
		print (ids[i]+','+str(Y[i]), file=f)
