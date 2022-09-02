import numpy as np
import os
import _pickle as cPickle
import collections
from e2efold.common.utils import *
from e2efold.common.long_seq_pre_post_process import *
from multiprocessing import Pool
from torch.utils import data
from collections import Counter
from random import shuffle
import torch

class RNASSDataGenerator(object):
    def __init__(self, data_dir, split, upsampling=False):
        self.data_dir = data_dir
        self.split = split
        self.upsampling = upsampling
        # Load vocab explicitly when needed
        self.load_data()
        # Reset batch pointer to zero
        self.batch_pointer = 0

    def load_data(self):
        p = Pool()
        data_dir = self.data_dir
        # Load the current split
        RNA_SS_data = collections.namedtuple('RNA_SS_data', 
            'seq ss_label length name pairs')
        with open(os.path.join(data_dir, '%s.pickle' % self.split), 'rb') as f:
            self.data = cPickle.load(f)
        if self.upsampling:
            self.data = self.upsampling_data()
        self.data_x = np.array([instance[0] for instance in self.data])
        self.data_y = np.array([instance[1] for instance in self.data])
        self.pairs = np.array([instance[-1] for instance in self.data])
        self.seq_length = np.array([instance[2] for instance in self.data])
        self.len = len(self.data)
        self.seq = list(p.map(encoding2seq, self.data_x))
        self.seq_max_len = len(self.data_x[0])
        # self.matrix_rep = np.array(list(p.map(creatmat, self.seq)))
        # self.matrix_rep = np.zeros([self.len, len(self.data_x[0]), len(self.data_x[0])])

    def upsampling_data(self):
        name = [instance.name for instance in self.data]
        d_type = np.array(list(map(lambda x: x.split('/')[2], name)))
        data = np.array(self.data)
        max_num = max(Counter(list(d_type)).values())
        data_list = list()
        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type==t)[0]
            data_list.append(data[index])
        final_d_list= list()
        # for d in data_list:
        #     index = np.random.choice(d.shape[0], max_num)
        #     final_d_list += list(d[index])
        for i in [0, 1, 5, 7]:
            d = data_list[i]
            index = np.random.choice(d.shape[0], max_num)
            final_d_list += list(d[index])

        for i in [2,3,4]:
            d = data_list[i]
            index = np.random.choice(d.shape[0], max_num*2)
            final_d_list += list(d[index])
        
        d = data_list[6]
        index = np.random.choice(d.shape[0], int(max_num/2))
        final_d_list += list(d[index])

        shuffle(final_d_list)
        return final_d_list


    def next_batch(self, batch_size):
        bp = self.batch_pointer
        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        batch_x = self.data_x[bp:bp + batch_size]
        batch_y = self.data_y[bp:bp + batch_size]
        batch_seq_len = self.seq_length[bp:bp + batch_size]

        self.batch_pointer += batch_size
        if self.batch_pointer >= len(self.data_x):
            self.batch_pointer = 0

        yield batch_x, batch_y, batch_seq_len

    def pairs2map(self, pairs):
        seq_len = self.seq_max_len
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact

    def next_batch_SL(self, batch_size):
        p = Pool()
        bp = self.batch_pointer
        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        data_y = self.data_y[bp:bp + batch_size]
        data_seq = self.data_x[bp:bp + batch_size]
        data_pairs = self.pairs[bp:bp + batch_size]

        self.batch_pointer += batch_size
        if self.batch_pointer >= len(self.data_x):
            self.batch_pointer = 0
        contact = np.array(list(map(self.pairs2map, data_pairs)))
        matrix_rep = np.zeros(contact.shape)
        yield contact, data_seq, matrix_rep

    def get_one_sample(self, index):

        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        data_y = self.data_y[index]
        data_seq = self.data_x[index]
        data_len = self.seq_length[index]
        data_pair = self.pairs[index]

        contact= self.pairs2map(data_pair)
        matrix_rep = np.zeros(contact.shape)
        return contact, data_seq, matrix_rep, data_len


    def random_sample(self, size=1):
        # random sample one RNA
        # return RNA sequence and the ground truth contact map
        index = np.random.randint(self.len, size=size)
        data = list(np.array(self.data)[index])
        data_seq = [instance[0] for instance in data]
        data_stru_prob = [instance[1] for instance in data]
        data_pair = [instance[-1] for instance in data]
        seq = list(map(encoding2seq, data_seq))
        contact = list(map(self.pairs2map, data_pair))
        return contact, seq, data_seq

    def get_one_sample_cdp(self, index):
        data_seq = self.data_x[index]
        data_label = self.data_y[index]

        return data_seq, data_label



# using torch data loader to parallel and speed up the data load process
class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data.get_one_sample(index)

class Dataset_1800(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        contacts, seq_embeddings, matrix_reps, seq_lens = self.data.get_one_sample(index)
        PE = get_pe(torch.Tensor([seq_lens]).long(), 1800).numpy()
        PE = torch.Tensor(PE[0]).float()
        small_seqs, comb_index_1 = get_chunk_combination(torch.Tensor(seq_embeddings).float())
        PE_small_seqs, comb_index_2 = get_chunk_combination(PE)
        contacts_b = get_chunk_gt(torch.Tensor(contacts).float(), comb_index_1)

        assert comb_index_1==comb_index_2

        seq_embedding_batch = torch.cat([seq.unsqueeze_(0) for seq in small_seqs], 0).float()
        PE_batch = torch.cat([pe.unsqueeze_(0) for pe in PE_small_seqs], 0).float()
        contacts_batch = torch.cat([contact.unsqueeze_(0) for contact in contacts_b], 0).float()

        return seq_embedding_batch, PE_batch, contacts_batch, comb_index_1, seq_embeddings, contacts, seq_lens



class Dataset_cdp(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data.get_one_sample_cdp(index)

    
