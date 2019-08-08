import pandas as pd
from tqdm import tqdm

pairLst = './fasion-resize-pairs-train.csv'
pairs_file_train = pd.read_csv(pairLst)
pairs = []
target_file_path = './fasion-resize-pairs-train-top.csv'

size = len(pairs_file_train)
for i in tqdm(range(size)):
    f, t = pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']
    if 'additional' not in f and 'full' not in f and 'additional' not in t and 'full' not in t:
    	pairs.append("{},{}\n".format(f, t))

with open(target_file_path, 'w+') as g:
	g.writelines(pairs)