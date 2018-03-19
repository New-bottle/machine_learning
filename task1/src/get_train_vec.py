from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import jieba
import numpy as np
import csv

data_path = '../data/'
train_ori = data_path + 'training.pk1'
train_vec = data_path + 'training_vec.pk1'
train_label_file = data_path + 'train.csv'

with open(train_ori, 'rb') as f:
	contents = pickle.load(f)
	titles = pickle.load(f)
	ids = pickle.load(f)
	labels = pickle.load(f)

def parse_data(ori_data):
    ans = []
    for obj in ori_data:
        line = obj
        seg_list = jieba.lcut(line.strip(), cut_all=False)
        ans.append(seg_list)
    return ans
data_set = parse_data(contents)

corpus = []
for item in data_set:
    corpus.append(" ".join(item))
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus)).toarray()

print (tfidf)
print ('111')

def get_vec(data_set):
    vec_list = []
    for each in data_set:
        this_vec = transformer.transform(vectorizer.transform([" ".join(each)])).toarray()
        vec_list.extend(this_vec)
    return vec_list

vec_list = get_vec(data_set)

with open(data_path + 'test_vec.pk1', 'wb') as f:
    pickle.dump(vec_list, f)
    pickle.dump(labels, f)
#   print (vec_list)
    f.close()

