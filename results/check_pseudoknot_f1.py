import os
import numpy as np
import _pickle as pickle
import pandas as pd

# for e2e
long_dir = './rnastralign/long/pseudoknot_ct'
short_dir = './rnastralign/short/pseudoknot_ct'

long_list = os.listdir(long_dir)
short_list = os.listdir(short_dir)
pred_list = list(filter(lambda x: 'pred' in x, long_list+short_list))
pseudoknot_f1 = list(map(lambda x: float(x.split('_')[0]), pred_list))
print('Exact f1: ', np.average(pseudoknot_f1))

# for rnastructure
with open('../data/rnastralign_test_pseudoknot_tag.pickle', 'rb') as f:
	pseudoknot_tag = pickle.load(f)
# load rna structure data
filepath = './traditional_method_results/rnastralign/results_no_shift/RNAStructure.tsv'
df = pd.read_csv(filepath, sep='\t', header=None)
rnastructure_f1 = df.iloc[:, -1].values

rnastructure_pse_f1 = list()
for i in range(len(pseudoknot_tag)):
	if pseudoknot_tag[i][-1]:
		rnastructure_pse_f1.append(rnastructure_f1[i])
