from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import jieba
import numpy as np
import csv

data_path = '../data/'
train_ori = data_path + 'training.pk1'
test_ori  = data_path + 'test.pk1'
train_vec = data_path + 'training_vec.pk1'
test_vec  = data_path + 'test_vec.pk1'
train_label_file = data_path + 'train.csv'

vectorizer = CountVectorizer(max_features = 2**13)
transformer = TfidfTransformer()

negmax = 26349
negnum = 0
posnum = 0
def parse_data(ori_data):
	ans = []
	for obj in ori_data:
		line = obj
		seg_list = jieba.lcut(line.strip(), cut_all=False)
		ans.append(" ".join(seg_list))
	return ans

def get_vec(data_set):
	vec_list = []
	for each in data_set:
		this_vec = transformer.transform(vectorizer.transform([" ".join(each)]))
		vec_list.extend(this_vec)
	return vec_list

if __name__=='__main__':
	with open(train_ori, 'rb') as f:
		contents = pickle.load(f)
		titles = pickle.load(f)
		ids = pickle.load(f)
		labels = pickle.load(f)

	data_set = parse_data(contents)

	corpus = []
	for item in data_set:
		corpus.append(" ".join(item))
	tfidf = transformer.fit_transform(vectorizer.fit_transform(data_set))
	print (tfidf)

#	vec_list = get_vec(data_set)
	vec_list = vectorizer.transform(data_set)
	print (vec_list.shape)

	with open(train_vec, 'wb') as f:
		pickle.dump(vec_list, f)
		pickle.dump(labels, f)
		f.close()
############################################################################################
	with open(test_ori, 'rb') as f:
		contents = pickle.load(f)
		titles = pickle.load(f)
		ids = pickle.load(f)
	print (len(contents))
	data_set = parse_data(contents)
#	vec_list = get_vec(data_set)
	vec_list = vectorizer.transform(data_set)
	print (vec_list.shape)

	with open(test_vec, 'wb') as f:
		pickle.dump(vec_list, f)
		f.close()
