import _pickle as cPickle
import numpy as np
import os
from os import walk
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../')
from common.utils import get_pairings
import seaborn as sns
import matplotlib.pyplot as plt

dataset = 'bpRNA_1m_90'
# ct_90_sim
datapath = './bpRNA_1m_90_BPSEQLFILES'
# datapath = './bpRNA/ct_90_sim'
length_limit = 600
# lower_limit = 600
seed = 0

# select all files within the preferred rna_type
file_list = list()

file_list = os.listdir(datapath)
file_list = list(map(lambda x: os.path.join(datapath, x), file_list))

# load data
data_list = list(map(lambda x: pd.read_csv(x, sep='\s+', skiprows=2,
        header=None) if 'CRW' in x else pd.read_csv(x, sep='\s+', skiprows=0,
        header=None) if 'PDB' in x or 'RFAM' in x else pd.read_csv(x, sep='\s+', skiprows=1,
        header=None), file_list))

# for 5s, the sequence length is from 102 to 135
seq_len_list= list(map(len, data_list))

file_length_dict = dict()
for i in range(len(seq_len_list)):
    file_length_dict[file_list[i]] = seq_len_list[i]

# print(len(seq_len_list))
# print(min(seq_len_list))
# print(max(seq_len_list))


# draw the squence length distribution
# fig, ax = plt.subplots(figsize=(9,5))
# sns.distplot(seq_len_list, kde=False, color='b')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel('Sequence length', fontsize=12)
# plt.ylabel('Count', fontsize=12)
# plt.savefig('bpRNA_1m_90.png',dpi=200,bbox_inches='tight')
# exit(1)

data_list = list(filter(lambda x: len(x)<=length_limit, data_list))
seq_len_list = list(map(len, data_list))
file_list = list(filter(lambda x: file_length_dict[x]<=length_limit, file_list))

print(len(seq_len_list))
print(min(seq_len_list))
print(max(seq_len_list))

def generate_label(data):
    rnadata1 = data.loc[:,0]
    rnadata2 = data.loc[:,2]
    rnastructure = []
    for i in range(len(rnadata2)):
        if rnadata2[i] == 0:
            rnastructure.append(".")
        else:
            if rnadata1[i] > rnadata2[i]:
                rnastructure.append(")")
            else:
                rnastructure.append("(")
    return ''.join(rnastructure)

def find_pseudoknot(data):
    rnadata1 = data.loc[:,0]
    rnadata2 = data.loc[:,2]
    flag = False
    for i in range(len(rnadata2)):
        for j in range(len(rnadata2)):
            if (rnadata1[i] < rnadata1[j] < rnadata2[i] < rnadata2[j]):
                flag = True
                break
    return flag

# generate the ".()" labeling for each position and the sequence
structure_list = list(map(generate_label, data_list))
seq_list = list(map(lambda x: ''.join(list(x.loc[:, 1])).upper(), data_list))
pairs_list = list(map(get_pairings, data_list))

# print(pairs_list[0])
# print(pairs_list[1])
# exit(1)


label_dict = {
    '.': np.array([1,0,0]),
    '(': np.array([0,1,0]),
    ')': np.array([0,0,1])
}
seq_dict = {
    'A':np.array([1,0,0,0]),
    'U':np.array([0,1,0,0]),
    'C':np.array([0,0,1,0]),
    'G':np.array([0,0,0,1]),
    'N':np.array([0,0,0,0]),
    'M':np.array([1,0,1,0]),
    'Y':np.array([0,1,1,0]),
    'W':np.array([1,0,0,0]),
    'V':np.array([1,0,1,1]),
    'K':np.array([0,1,0,1]),
    'R':np.array([1,0,0,1]),
    'I':np.array([0,0,0,0]),
    'X':np.array([0,0,0,0]),
    'S':np.array([0,0,1,1]),
    'D':np.array([1,1,0,1]),
    'P':np.array([0,0,0,0]),
    'B':np.array([0,1,1,1]),
    'H':np.array([1,1,1,0]),
    '.':np.array([0,0,0,0]),
    '~':np.array([0,0,0,0]),
    '_':np.array([0,0,0,0])
}

def seq_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: seq_dict[x], str_list))
    # need to stack
    return np.stack(encoding, axis=0)

def stru_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: label_dict[x], str_list))
    # need to stack
    return np.stack(encoding, axis=0)

def padding(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0,maxlen-a),(0,0)), 'constant')

# label and sequence encoding, plus padding to the maximum length
max_len = max(seq_len_list)
seq_encoding_list = list(map(seq_encoding, seq_list))
stru_encoding_list = list(map(stru_encoding, structure_list))

seq_encoding_list_padded = list(map(lambda x: padding(x, max_len),
    seq_encoding_list))
stru_encoding_list_padded = list(map(lambda x: padding(x, max_len),
    stru_encoding_list))

# gather the information into a list of tuple
RNA_SS_data = collections.namedtuple('RNA_SS_data', 
    'seq ss_label length name pairs')
RNA_SS_data_list = list()
for i in range(len(data_list)):
    RNA_SS_data_list.append(RNA_SS_data(seq=seq_encoding_list_padded[i],
        ss_label=stru_encoding_list_padded[i], 
        length=seq_len_list[i], name=file_list[i], pairs=pairs_list[i]))

## training test split
RNA_SS_train, RNA_SS_test = train_test_split(RNA_SS_data_list, 
    test_size=0.2, random_state=seed)

RNA_SS_test, RNA_SS_val = train_test_split(RNA_SS_test, 
    test_size=0.5, random_state=seed)

# savepath = dataset+"_"+"_".join(rna_type)
savepath = dataset+"_" + str(length_limit)

os.mkdir(savepath)

for i in ['train', 'test', 'val']:
    with open(savepath+'/%s.pickle' % i, 'wb') as f:
        cPickle.dump(eval('RNA_SS_'+i), f)

