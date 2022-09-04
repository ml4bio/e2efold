#!/usr/bin/env python
import _pickle as cPickle
import numpy as np
import os
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from multiprocessing import Pool


dataset = 'archiveII'
rna_type = '5s'
datapath = './archiveII'
seed = 0

# select all the 5s files
file_list = os.listdir(datapath)
file_list = list(filter(lambda x: not x.startswith(rna_type) and x.endswith(".ct"), 
    file_list))

# load data, 5s do not have pseudoknot so we do not have to delete them
data_list = list()
for file in file_list:
    df = pd.read_csv(os.path.join(datapath, file), sep='\s+', skiprows=1,
        header=None)
    data_list.append(df)



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

p = Pool()
pseudoknot_check = np.array(list(p.map(find_pseudoknot, data_list)))
index = np.where(pseudoknot_check)

pseudoknot_tag = list(zip(list(file_list), list(pseudoknot_check)))

with open('archiveII_pseudoknot_tag.pickle', 'wb') as f:
    cPickle.dump(pseudoknot_tag, f)