import _pickle as cPickle
import numpy as np
import os
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from e2efold.common.utils import get_pairings

dataset = 'archiveII'
rna_type = ['5s', '16s', '23s', 'grp1', 'grp2', 'RNaseP', 
    'srp', 'telomerase', 'tmRNA', 'tRNA']
datapath = './archiveII'
seed = 0

# for rna_type in rna_types:
# select all the 5s files
file_list = os.listdir(datapath)
file_list = list(filter(lambda x: x.startswith(tuple(rna_type)) and x.endswith(".ct"), 
    file_list))

# load data, 5s do not have pseudoknot so we do not have to delete them
data_list = list()
for file in file_list:
    df = pd.read_csv(os.path.join(datapath, file), sep='\s+', skiprows=1,
        header=None)
    data_list.append(df)

# for 5s, the sequence length is from 102 to 135
seq_len_list= list(map(len, data_list))
print(rna_type)
print(min(seq_len_list))
print(max(seq_len_list))
print(len(seq_len_list))

def generate_label(data):
    rnadata1 = data.loc[:,0]
    rnadata2 = data.loc[:,4]
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
    rnadata2 = data.loc[:,4]
    flag = False
    for i in range(len(rnadata2)):
        for j in range(len(rnadata2)):
            if (rnadata1[i] < rnadata1[j] < rnadata2[i] < rnadata2[j]):
                flag = True
                break
    return flag

# generate the ".()" labeling for each position and the sequence
structure_list = list(map(generate_label, data_list))
seq_list = list(map(lambda x: ''.join(list(x.loc[:, 1])), data_list))
pairs_list = list(map(get_pairings, data_list))

label_dict = {
    '.': np.array([1,0,0]), 
    '(': np.array([0,1,0]), 
    ')': np.array([0,0,1])
}
seq_dict = {
    'A':np.array([1,0,0,0]),
    'U':np.array([0,1,0,0]),
    'C':np.array([0,0,1,0]),
    'G':np.array([0,0,0,1])
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

savepath = dataset+"_"+"_".join(rna_type)
# savepath = dataset+"_all"
os.mkdir(savepath)

for i in ['train', 'test', 'val']:
    with open(savepath+'/%s.pickle' % i, 'wb') as f:
        cPickle.dump(eval('RNA_SS_'+i), f)

