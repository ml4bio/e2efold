#!/usr/bin/env python
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import math
import torch
import torch.nn as nn
from scipy import signal
from multiprocessing import Pool
from functools import partial
import torch.nn.functional as F
import argparse
import pandas as pd
from scipy.sparse import diags
import random
import os

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
    'H':np.array([1,1,1,0])
}

char_dict = {
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G'
}


def get_args():
    argparser = argparse.ArgumentParser(description="diff through pp")
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='./config.json',
        help='The Configuration file'
    )
    argparser.add_argument('--test', type=bool, default=False, 
        help='skip training to test directly.')
    args = argparser.parse_args()
    return args

def soft_sign(x, k):
    return torch.sigmoid(k * x)

def seq_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: seq_dict[x], str_list))
    # need to stack
    return np.stack(encoding, axis=0)
    
def Gaussian(x):
    return math.exp(-0.5*(x*x))

def paired(x,y):
    if x == 'A' and y == 'U':
        return 2
    elif x == 'G' and y == 'C':
        return 3
    elif x == 'G' and y == 'U':
        return 0.8
    elif x == 'U' and y == 'A':
        return 2
    elif x == 'C' and y == 'G':
        return 3
    elif x == 'U' and y == 'G':
        return 0.8
    else:
        return 0

def creatmat(data):
    mat = np.zeros([len(data),len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add <len(data):
                    score = paired(data[i - add],data[j + add])
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1,30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(data[i + add],data[j - add])
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * Gaussian(add)
                    else:
                        break
            mat[[i],[j]] = coefficient
    return mat

def createzeromat(data):
    mat = np.zeros([len(data),len(data)])
    return mat

# can't deal with pseudoknot
def ct2struct(ct):
	stack = list()
	struct = list()
	for i in range(len(ct)):
		if ct[i]=='(':
			stack.append(i)
		if ct[i]==')':
			left = stack.pop()
			struct.append([left, i])
	return struct

# only consider the lower triangle matrix
# can't deal with pseudoknot
def prob2map(prob):
	contact = np.zeros([len(prob), len(prob)])
	left_index = np.where(prob[:, 1])[0]
	right_index = np.where(prob[:, 2])[0][::-1]
	ct = np.array(['.']*len(prob))
	ct[left_index] = '('
	ct[right_index] = ')'
	struct = ct2struct(ct)
	for index in struct:
		index_1 = max(index[0], index[1])
		index_2 = min(index[0], index[1])
		contact[index_1, index_2]=1
	triu_index = np.triu_indices(len(contact),k=0)
	contact[triu_index]=-1
	return contact

def contact2sym(contact):
    triu_index = np.triu_indices(contact.shape[-1],k=0)
    contact[triu_index]=0
    return contact+np.transpose(contact)

# can't deal with pseudoknot
def prob2struct(prob):
	left_index = np.where(prob[:, 1])[0]
	right_index = np.where(prob[:, 2])[0][::-1]
	ct = np.array(['.']*len(prob))
	ct[left_index] = '('
	ct[right_index] = ')'
	struct = ct2struct(ct)
	return struct


def encoding2seq(arr):
	seq = list()
	for arr_row in list(arr):
		if sum(arr_row)==0:
			seq.append('.')
		else:
			seq.append(char_dict[np.argmax(arr_row)])
	return ''.join(seq)


def contact2ct(contact, sequence_encoding, seq_len):
    seq = encoding2seq(sequence_encoding)[:seq_len].replace('.', 'N')
    contact = contact[:seq_len, :seq_len]
    structure = np.where(contact)
    pair_dict = dict()
    for i in range(seq_len):
        pair_dict[i] = -1
    for i in range(len(structure[0])):
        pair_dict[structure[0][i]] = structure[1][i]
    first_col = list(range(1, seq_len+1))
    second_col = list(seq)
    third_col = list(range(seq_len))
    fourth_col = list(range(2, seq_len+2))
    fifth_col = [pair_dict[i]+1 for i in range(seq_len)]
    last_col = list(range(1, seq_len+1))
    df = pd.DataFrame()
    df['index'] = first_col
    df['base'] = second_col
    df['index-1'] = third_col
    df['index+1'] = fourth_col
    df['pair_index'] = fifth_col
    df['n'] = last_col
    return df


    

def padding(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0,maxlen-a),(0,0)), 'constant')

def F1_low_tri(opt_state, true_a):
	tril_index = np.tril_indices(len(opt_state),k=-1)
	return f1_score(true_a[tril_index], opt_state[tril_index])

def acc_low_tri(opt_state, true_a):
	tril_index = np.tril_indices(len(opt_state),k=-1)
	return accuracy_score(true_a[tril_index], opt_state[tril_index])



def logit2binary(pred_contacts):
    sigmoid_results = torch.sigmoid(pred_contacts)
    binary = torch.where(sigmoid_results > 0.5, 
        torch.ones(pred_contacts.shape), 
        torch.zeros(pred_contacts.shape))
    return binary


def unravel2d_torch(ind, ncols):
    x = ind / ncols 
    y = ind % ncols
    return (int(x),int(y))


def postprocess_sort(contact):
    ncols = contact.shape[-1]
    contact = torch.sigmoid(contact)
    contact_flat = contact.reshape(-1)
    final_contact = torch.zeros(contact.shape)
    contact_sorted, sorted_ind = torch.sort(contact_flat,descending=True)
    ind_one = sorted_ind[contact_sorted>0.9]
    length = len(ind_one)
    use = min(length, 10000)
    ind_list = list(map(lambda x: unravel2d_torch(x, ncols), ind_one[:use]))
    row_list = list()
    col_list = list()
    for ind_x, ind_y in ind_list:
        if (ind_x not in row_list) and (ind_y not in col_list):
            row_list.append(ind_x)
            col_list.append(ind_y)
            final_contact[ind_x, ind_y] = 1
        else:
            ind_list.remove((ind_x, ind_y))
    return final_contact

def conflict_sort(contacts):
    processed = list(map(postprocess_sort, 
        list(contacts)))
    return processed


def check_thredhold(pred_contacts, contacts):
    a = contacts[0]
    b = pred_contacts[0]
    b = torch.sigmoid(b)
    b = b.cpu().numpy()
    print(len(np.where(a>0)[0]))
    print(len(np.where(b>0.5)[0]))
    print(min(b[np.where(a>0)]))
    print(len(np.where(b> min(b[np.where(a>0)]))[0]))

def postprocess_sampling(contact):
    from scipy.special import softmax
    ncols = contact.shape[-1]
    contact = torch.sigmoid(contact)
    contact_flat = contact.reshape(-1)
    final_contact = torch.zeros(contact.shape)
    contact_sorted, sorted_ind = torch.sort(contact_flat,descending=True)
    ind_one = sorted_ind[contact_sorted>0.5]
    used_values = contact_sorted[contact_sorted>0.5]
    ind_list = list(map(lambda x: unravel2d_torch(x, ncols), ind_one))
    row_list = list()
    col_list = list()
    prob = used_values.cpu().numpy()
    # for each step, sample one from the list
    # then remove that value and the index, add it into the row and col list
    # before add check the row and col list first, to avoid conflict
    for i in range(len(used_values)):
        prob = softmax(prob)
        ind = int(np.random.choice(len(prob), 1, p=prob))
        ind_x, ind_y = ind_list[ind]
        if (ind_x not in row_list) and (ind_y not in col_list):
            row_list.append(ind_x)
            col_list.append(ind_y)
            final_contact[ind_x, ind_y] = 1
        ind_list.remove((ind_x, ind_y))
        prob = np.delete(prob, ind)
    return final_contact

def conflict_sampling(contacts):
    processed = list(map(postprocess_sampling, 
        list(contacts)))
    return processed


# we first apply a kernel to the ground truth a
# then we multiple the kernel with the prediction, to get the TP allows shift
# then we compute f1
# we unify the input all as the symmetric matrix with 0 and 1, 1 represents pair
def evaluate_shifted(pred_a, true_a):
    kernel = np.array([[0.0,1.0,0.0],
                        [1.0,1.0,1.0],
                        [0.0,1.0,0.0]])
    pred_a_filtered = signal.convolve2d(pred_a, kernel, 'same')
    fn = len(torch.where((true_a - torch.Tensor(pred_a_filtered))==1)[0])
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    tp = true_p - fn
    fp = pred_p - tp
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1_score = 2*tp/(2*tp + fp + fn)
    return precision, recall, f1_score

def evaluate_exact(pred_a, true_a):
    tp_map = torch.sign(torch.Tensor(pred_a)*torch.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1_score = 2*tp/(2*tp + fp + fn)
    return precision, recall, f1_score

def test_evaluation():
    pred_a = np.zeros([4,5])
    true_a = np.zeros([4,5])
    true_a[0,1]=1;true_a[1,1]=1;true_a[2,2]=1;true_a[3,3]=1
    pred_a[0,2]=1;pred_a[1,2]=1;pred_a[2,0]=1;pred_a[3,3]=1;pred_a[3,1]=1
    print(evaluate_shifted(pred_a, true_a))
    print(evaluate_exact(pred_a, true_a))

def constraint_matrix(x):
    base_a = x[:, 0]
    base_u = x[:, 1]
    base_c = x[:, 2]
    base_g = x[:, 3]
    au = torch.matmul(base_a.view(-1, 1), base_u.view(1, -1))
    au_ua = au + au.t()
    cg = torch.matmul(base_c.view(-1, 1), base_g.view(1, -1))
    cg_gc = cg + cg.t()
    ug = torch.matmul(base_u.view(-1, 1), base_g.view(1, -1))
    ug_gu = ug + ug.t()
    return au_ua + cg_gc + ug_gu

def constraint_matrix_batch(x):
    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    return au_ua + cg_gc + ug_gu


def constraint_matrix_batch_diag(x, offset=3):
    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    m = au_ua + cg_gc + ug_gu

    # create the band mask
    mask = diags([1]*7, [-3, -2, -1, 0, 1, 2, 3], 
        shape=(m.shape[-2], m.shape[-1])).toarray()
    m = m.masked_fill(torch.Tensor(mask).bool(), 0)
    return m

def contact_map_masks(seq_lens, max_len):
    n_seq = len(seq_lens)
    masks = np.zeros([n_seq, max_len, max_len])
    for i in range(n_seq):
        l = int(seq_lens[i].cpu().numpy())
        masks[i, :l, :l]=1
    return masks

# for test the f1 loss filter
# true_a = torch.Tensor(np.arange(25)).view(5,5).unsqueeze(0)

def f1_loss(pred_a, true_a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_a  = -(F.relu(-pred_a+1)-1)

    true_a = true_a.unsqueeze(1)
    unfold = nn.Unfold(kernel_size=(3, 3), padding=1)
    true_a_tmp = unfold(true_a)
    w = torch.Tensor([0, 0.0, 0, 0.0, 1, 0.0, 0, 0.0, 0]).to(device)
    true_a_tmp = true_a_tmp.transpose(1, 2).matmul(w.view(w.size(0), -1)).transpose(1, 2)
    true_a = true_a_tmp.view(true_a.shape)
    true_a = true_a.squeeze(1)

    tp = pred_a*true_a
    tp = torch.sum(tp, (1,2))

    fp = pred_a*(1-true_a)
    fp = torch.sum(fp, (1,2))

    fn = (1-pred_a)*true_a
    fn = torch.sum(fn, (1,2))

    f1 = torch.div(2*tp, (2*tp + fp + fn))
    return 1-f1.mean()    


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

# return index of contact pairing, index start from 0
def get_pairings(data):
    rnadata1 = list(data.loc[:,0].values)
    rnadata2 = list(data.loc[:,4].values)
    rna_pairs = list(zip(rnadata1, rnadata2))
    rna_pairs = list(filter(lambda x: x[1]>0, rna_pairs))
    rna_pairs = (np.array(rna_pairs)-1).tolist()
    return rna_pairs


def generate_label_dot_bracket(data):
    rnadata1 = data.loc[:,0]
    rnadata2 = data.loc[:,4]
    rnastructure = []
    for i in range(len(rnadata2)):
        if rnadata2[i] <= 0:
            rnastructure.append(".")
        else:
            if rnadata1[i] > rnadata2[i]:
                rnastructure.append(")")
            else:
                rnastructure.append("(")
    return ''.join(rnastructure)


# extract the pseudoknot index given the data
def extract_pseudoknot(data):
    rnadata1 = data.loc[:,0]
    rnadata2 = data.loc[:,4]
    for i in range(len(rnadata2)):
        for j in range(len(rnadata2)):
            if (rnadata1[i] < rnadata1[j] < rnadata2[i] < rnadata2[j]):
                print(i,j)
                break

def get_pe(seq_lens, max_len):
    num_seq = seq_lens.shape[0]
    pos_i_abs = torch.Tensor(np.arange(1,max_len+1)).view(1, 
        -1, 1).expand(num_seq, -1, -1).double()
    pos_i_rel = torch.Tensor(np.arange(1,max_len+1)).view(1, -1).expand(num_seq, -1)
    pos_i_rel = pos_i_rel.double()/seq_lens.view(-1, 1).double()
    pos_i_rel = pos_i_rel.unsqueeze(-1)
    pos = torch.cat([pos_i_abs, pos_i_rel], -1)

    PE_element_list = list()
    # 1/x, 1/x^2
    PE_element_list.append(pos)
    PE_element_list.append(1.0/pos_i_abs)
    PE_element_list.append(1.0/torch.pow(pos_i_abs, 2))

    # sin(nx)
    for n in range(1, 50):
        PE_element_list.append(torch.sin(n*pos))

    # poly
    for i in range(2, 5):
        PE_element_list.append(torch.pow(pos_i_rel, i))

    for i in range(3):
        gaussian_base = torch.exp(-torch.pow(pos, 
            2))*math.sqrt(math.pow(2,i)/math.factorial(i))*torch.pow(pos, i)
        PE_element_list.append(gaussian_base)

    PE = torch.cat(PE_element_list, -1)
    for i in range(num_seq):
        PE[i, seq_lens[i]:, :] = 0
    return PE

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True






