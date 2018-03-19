import jieba
import pickle
import json
import pandas as pd
train_file = "../data/output.json"
data_path = '../data/'

bag = []
with open(train_file, 'r') as f:
	lines = f.readlines()
	for line in lines:
		tmp = json.loads(line)

for seg in contents:
	seg_list = jieba.lcut(seg.strip(), cut_all=False)
	bag.extend(seg_list)

with open('../data/remove.txt', 'r') as f:
	punctuation = f.read().split('\n')


cleanbag = []
for token in bag:
	for punc in punctuation:
		token = token.replace(punc, '')
		if token != "":
			cleanbag.append(token)

dic = list(set(cleanbag))

counts = pd.Series(cleanbag).value_counts()

with open(data_path + 'dict.txt', 'w') as f:
	for index, each in enumerate(counts):
		if each > 0:
			print (str(counts.index[index]), file = f)

pd.set_option('display.max_rows', None)
with open(data_path + 'dict_count.txt', 'w') as f:
	print (str(counts), file = f)
