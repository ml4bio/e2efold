import _pickle as cPickle
import numpy as np
import os
from os import walk
import pandas as pd
import collections
from collections import defaultdict
from e2efold.common.utils import get_pairings
import seaborn as sns
import matplotlib.pyplot as plt

dataset = 'rnastralign'
rna_types = ['tmRNA', 'tRNA', 'telomerase', 'RNaseP', 
    'SRP', '16S_rRNA', '5S_rRNA', 'group_I_intron']

# rna_types = ['5S_rRNA', '16S_rRNA']
datapath = './RNAStrAlign'
seed = 0

# select all files within the preferred rna_type
file_list = list()

for rna_type in rna_types:
    type_dir = os.path.join(datapath, rna_type+'_database')
    # print(type_dir)
    for r, d, f in walk(type_dir):
        for file in f:
            if file.endswith(".ct"):
                file_list.append(os.path.join(r,file))



# load data
data_list = list(map(lambda x: pd.read_csv(x, sep='\s+', skiprows=1,
        header=None), file_list))

seq_list = list(map(lambda x: ''.join(list(x.loc[:, 1])), data_list))

seq_file_pair_list = list(zip(seq_list, file_list))
d = defaultdict(list)
for k,v in seq_file_pair_list:
	d[k].append(v)
unique_seqs = list()
seq_files = list()
for k,v in d.items():
	unique_seqs.append(k)
	seq_files.append(v)

original_seq_len = list(map(len, seq_list))
unique_seq_len = list(map(len, unique_seqs))
cluster_size = list(map(len, seq_files))
used_files = list(map(lambda x: x[0], seq_files))
used_files_rna_type = list(map(lambda x: x.split('/')[2], used_files))

# draw the squence length distribution
fig, ax = plt.subplots(figsize=(9,5))
sns.distplot(unique_seq_len, kde=False, color='b')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Sequence length', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.savefig('RNAStrAlign_dis.png',dpi=200,bbox_inches='tight')

# check the testing data
RNA_SS_data = collections.namedtuple('RNA_SS_data', 
    'seq ss_label length name pairs')
with open('rnastralign_all/test.pickle', 'rb') as f:
	test_all_600 = cPickle.load(f)
with open('rnastralign_all/train.pickle', 'rb') as f:
	train_all_600 = cPickle.load(f)

file_seq_d = dict()
for k,v in seq_file_pair_list:
	file_seq_d[v] =k

train_files = [instance.name for instance in train_all_600]
train_seqs = [file_seq_d[file] for file in train_files]
train_in_files = list()
for seq in train_seqs:
	files_tmp = d[seq]
	train_in_files += files_tmp
train_in_files = list(set(train_in_files))

test_files = [instance.name for instance in test_all_600]
test_set = list(set(test_files) - set(test_files).intersection(train_in_files))
test_seqs = [file_seq_d[file] for file in test_set]
test_seq_file_pair_list = zip(test_seqs, test_set)
test_seq_file_d = defaultdict(list)
for k,v in test_seq_file_pair_list:
	test_seq_file_d[k].append(v)
test_files_used = [test_seq_file_d[seq][0] for seq in test_seqs]
test_rna_type = list(map(lambda x: x.split('/')[2], test_files_used))

# use the test_files_used to filter the test files
test_all_600_used = list()
for instance in test_all_600:
	if instance.name in test_files_used:
		test_all_600_used.append(instance)

with open('rnastralign_all/test_no_redundant.pickle', 'wb') as f:
	cPickle.dump(test_all_600_used,f)

test_16s = list()
for instance in test_all_600_used:
	if '16S_rRNA' in instance.name:
		test_16s.append(instance)

with open('rnastralign_all/test_16s.pickle', 'wb') as f:
	cPickle.dump(test_16s,f)
