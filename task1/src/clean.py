import json
import re
path = "./data/"
train_data_file = "simple.json"


with open(path+train_data_file, 'r') as f:
	for line in f:
		print (json.loads(re.compile(r'<[^>]+>',re.S).sub('',line)))

'''
with open(path+train_data_file, 'r') as f:
	with open(path+"train_data.txt", 'w') as out:
		for line in f:
			out.write (re.compile(r'<[^>]+>',re.S).sub('',line))
'''				
