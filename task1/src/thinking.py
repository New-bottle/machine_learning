import pickle
train_file = '../data/training.pk1'

with open(train_file, 'rb') as f:
	contents = pickle.load(f)
	titles = pickle.load(f)
	ids = pickle.load(f)

print ('\n'.join(titles))
