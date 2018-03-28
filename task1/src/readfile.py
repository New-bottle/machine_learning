#!/usr/bin/python3
import pickle
import csv
import jieba
import re
import json
path = "../data/"
train_data_file = 'simple.json'
train_data_file = "train.json"
train_label_file = path+'train.csv'

def read_data(i_filename):
	X = []
	with open(i_filename, 'r') as f:
		for line in f:
			line = re.compile(r'<[^>]+>',re.S).sub('',line) # remove <html>
			tmp = json.loads(line)
			tmp["content"] = tmp["content"].replace("\u3000", '')
			tmp["content"] = tmp["content"].replace("\xa0", "")
			X.append(tmp)
		'''
		with open(path+"output.json", 'w') as out:
			for line in X:
				line['content'] = re.compile(r'([\d]+)',re.S).sub('',line['content']) # remove numbers
				print(json.dumps(line), file=out)
		'''
	print (len(X))
	return X



if __name__ == '__main__':
	contents = []
	titles = []
	ids = []
	Y = []
	X = read_data(path+train_data_file)

	for line in X:
		contents.append(line['content'])
		titles.append(line['title'])
		ids.append(line['id'])

	with open(train_label_file, 'r') as f:
		csvreader = csv.DictReader(f)
		for line in csvreader:
			Y.append(line['pred'])

#	Y = Y[0:1000]
	with open(path + 'training.pk1', 'wb') as f:
		pickle.dump(contents, f)
		pickle.dump(titles, f)
		pickle.dump(ids, f)
		pickle.dump(Y, f)

	X = read_data(path+'test.json')
#	X = X[0:1000]
	contents = []
	titles = []
	ids = []
	for line in X:
		contents.append(line['content'])
		titles.append(line['title'])
		ids.append(line['id'])
	with open(path + 'test.pk1', 'wb') as f:
		pickle.dump(contents, f)
		pickle.dump(titles, f)
		pickle.dump(ids, f)
