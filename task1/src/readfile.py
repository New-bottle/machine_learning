#!/usr/bin/python3
import pickle
import csv
import jieba
import re
import json
path = "../data/"
train_data_file = "train.json"
train_data_file = "simple.json"
train_label_file = path+'train.csv'

contents = []
titles = []
ids = []

X = []
with open(path+train_data_file, 'r') as f:
	for line in f:
		line = re.compile(r'<[^>]+>',re.S).sub('',line) # remove <html>
		tmp = json.loads(line)
		tmp["content"] = tmp["content"].replace("\u3000", '')
		tmp["content"] = tmp["content"].replace("\xa0", "")
		X.append(tmp)
	with open(path+"output.json", 'w') as out:
		for line in X:
			line['content'] = re.compile(r'([\d]+)',re.S).sub('',line['content']) # remove numbers
			print(json.dumps(line), file=out)

for line in X:
	contents.append(line['content'])
	titles.append(line['title'])
	ids.append(line['id'])

Y = []
with open(train_label_file, 'r') as f:
	csvreader = csv.DictReader(f)
	for line in csvreader:
		Y.append(line['pred'])

with open(path + 'training.pk1', 'wb') as f:
	pickle.dump(contents, f)
	pickle.dump(titles, f)
	pickle.dump(ids, f)
	pickle.dump(Y, f)
