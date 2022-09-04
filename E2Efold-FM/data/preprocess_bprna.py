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
import pandas as pd

# there is something wrong with this data, how to process
def generate_label(data):
    rnadata1 = data.loc[:,0]
    rnadata2 = data.loc[:,4]  #data.loc[:,2]
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


dataset = 'SPOT_RNA_500'
set_name = "test"
ann_file = os.path.join("/share/liyu/RNA/Data/SPOT-RNA/preprocessed/bpRNA/ann", set_name + ".csv")


df = pd.read_csv(ann_file)
target_names = df["filename"].values

length_limit = 500
max_len = length_limit
seed = 0

save_dir = os.path.join("/user/liyu/cjy/RNA/methods/E2Efold-FM/data/", dataset + "_" + str(length_limit))
if os.path.exists(save_dir) != True:
    os.mkdir(save_dir)


# rna-fm prob map
ss_prob_dir = "/user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/ss-rnafm/r-ss"
ss_probmap_list = list(map(lambda x: np.load(os.path.join(ss_prob_dir, x+".npy")), target_names))

def padding2d(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0,maxlen-a),(0,maxlen-a)), 'constant')
ss_probmap_list_padded = list(map(lambda x: padding2d(x, max_len), ss_probmap_list))


# load ct files
ct_file_dir = "/share/liyu/RNA/Data/SPOT-RNA/preprocessed/bpRNA/ct"

ct_file_names = [name + ".ct" for name in target_names] #os.listdir(ct_file_dir)
ct_file_list = list(map(lambda x: os.path.join(ct_file_dir, x), ct_file_names))


#tf = '/share/liyu/RNA/Data/SPOT-RNA/preprocessed/bpRNA/ct/bpRNA_RFAM_37340.ct'
#tf_df = pd.read_csv(tf, sep='\s+', skiprows=0, header=None)


# load data
ct_data_list = list(map(lambda x: pd.read_csv(x, sep='\s+', skiprows=1, header=None), ct_file_list))
#ct_data_list = list(
#    map(lambda x: pd.read_csv(x, sep='\s+', skiprows=2, header=None) if 'CRW' in x
#    else pd.read_csv(x, sep='\s+', skiprows=0, header=None) if 'PDB' in x or 'RFAM' in x
#    else pd.read_csv(x, sep='\s+', skiprows=1, header=None), ct_file_list))

length_list = list(map(len, ct_data_list))

# create a name-length dict
file_length_dict = dict()
for i in range(len(length_list)):
    file_length_dict[ct_file_list[i]] = length_list[i]

# filter data
ct_data_list = list(filter(lambda x: len(x) <= length_limit, ct_data_list))
length_list = list(map(len, ct_data_list))
ct_file_list = list(filter(lambda x: file_length_dict[x] <= length_limit, ct_file_list))

print("length Num:{}, Min:{}, Max:{}".format(len(length_list), min(length_list), max(length_list)))

# generate the ".()" labeling for each position and the sequence
structure_list = list(map(generate_label, ct_data_list))
seq_list = list(map(lambda x: ''.join(list(x.loc[:, 1])).upper(), ct_data_list))
pairs_list = list(map(get_pairings, ct_data_list))


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

def padding(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0,maxlen-a),(0,0)), 'constant')




# label and sequence encoding, plus padding to the maximum length
max_len = length_limit   # max(length_list)
seq_encoding_list = list(map(seq_encoding, seq_list))
stru_encoding_list = list(map(stru_encoding, structure_list))

seq_encoding_list_padded = list(map(lambda x: padding(x, max_len), 
    seq_encoding_list))
stru_encoding_list_padded = list(map(lambda x: padding(x, max_len), 
    stru_encoding_list))

# gather the information into a list of tuple
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs embedding')
RNA_SS_data_list = list()
for i in range(len(ct_data_list)):
    RNA_SS_data_list.append(
        RNA_SS_data(
            seq=seq_encoding_list_padded[i], ss_label=stru_encoding_list_padded[i],
            length=length_list[i], name=ct_file_list[i], pairs=pairs_list[i],
            embedding=ss_probmap_list_padded[i],
        )
    )

with open(save_dir + '/%s.pickle' % set_name, 'wb') as f:
    cPickle.dump(RNA_SS_data_list, f)


