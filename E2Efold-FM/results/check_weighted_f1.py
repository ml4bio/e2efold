import _pickle as pickle
import pandas as pd
import os
import numpy as np

with open('rnastralign_long_prediction_dict.pickle', 'rb') as f:
	ct_dict = pickle.load(f)
names = list(ct_dict.keys())

f1 = [ct_dict[name]['f1'] for name in names]
seq_len = [len(ct_dict[name]['true_ct']) for name in names]

weighted_f1 = np.sum(np.array(f1)*np.array(seq_len)/np.sum(seq_len))


# weighted f1 short
# 0.6796

# weighted f1 long 
# 0.7902