import _pickle as pickle
import pandas as pd
import os
from multiprocessing import Pool
from sklearn.metrics import confusion_matrix 


with open('rnastralign_short_prediction_dict.pickle', 'rb') as f:
	ct_dict_short = pickle.load(f)
with open('rnastralign_long_prediction_dict.pickle', 'rb') as f:
	ct_dict = pickle.load(f)

ct_dict.update(ct_dict_short)

def find_pseudoknot(data):
    rnadata1 = data.iloc[:,0]
    rnadata2 = data.iloc[:,4]
    flag = False
    for i in range(len(rnadata2)):
        for j in range(len(rnadata2)):
            if (rnadata1[i] < rnadata1[j] < rnadata2[i] < rnadata2[j]):
                flag = True
                break
    return flag

name_list = list(ct_dict.keys())
ct_list = list(map(lambda x: ct_dict[x]['pred_ct'], name_list))
p = Pool()
pseudo_tag = p.map(find_pseudoknot, ct_list)

pseudo_tuple_list = list(zip(name_list, pseudo_tag))

with open('e2e_pred_pseudoknot_tag.pickle', 'wb') as f:
    pickle.dump(pseudo_tuple_list, f)

# get confusion matrix

with open('../data/rnastralign_test_pseudoknot_tag.pickle', 'rb') as f:
	true_tag = pickle.load(f)

pseudoknot_dict = dict()
for name, tag in true_tag:
	pseudoknot_dict[name] = tag

ground_truth_label = list(map(lambda x: pseudoknot_dict[x], 
	name_list))
pred_label = pseudo_tag
confusion_matrix(ground_truth_label, pred_label)