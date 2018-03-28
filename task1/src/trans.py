import csv
with open('submit.csv', 'r') as f:
	csvreader = csv.DictReader(f)
	with open('../data/LR.csv', 'w') as fout:
		print ('id,pred', file = fout)
		for line in csvreader:
			print (str(line['id']) + ',' + str(1.0-float(line['pred'])), file = fout)


