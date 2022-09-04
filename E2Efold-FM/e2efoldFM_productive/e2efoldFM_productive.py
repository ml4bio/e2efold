import torch.optim as optim
import sys
sys.path.append('../')
from torch.utils import data
import shutil

from e2efoldFM.models import ContactNetwork, ContactNetwork_test, ContactNetwork_fc
from e2efoldFM.models import ContactAttention, ContactAttention_simple_fix_PE
from e2efoldFM.models import Lag_PP_NN, RNA_SS_e2e, Lag_PP_zero, Lag_PP_perturb
from e2efoldFM.models import Lag_PP_mixed, ContactAttention_simple, No_Noise
#from e2efoldFM.models import NoNoise
from e2efoldFM.new_common.utils import *
from e2efoldFM.new_common.config import process_config
from e2efoldFM.evaluation import all_test_only_e2e

import os
import requests
import zipfile
import glob
from Bio import SeqIO

args = get_args()

URL = "https://proj.cse.cuhk.edu.hk/rnafm/api/predict_all"
def convert_to_fasta(fileName):
    fileSeq = open(fileName, 'r')
    with open(fileName+'.fasta', 'w') as fileF:
        fileF.write('>' + fileName + '\n')
        fileF.write(fileSeq.read())


def get_rna_ss(filename):
    file = {'predict_all_file': open(filename+'.fasta', 'rb')}
    params = {'date_id': '1'}
    # sending get request and saving the response as response object
    r = requests.post(url=URL, files=file, params=params)
    res = requests.get(url=URL, params={'filename': filename+'.fasta', 'date_id': '1'})

    with open(filename+'.zip', 'wb') as outFile:
        outFile.write(res.content)
    with zipfile.ZipFile(filename+".zip", "r") as zip_ref:
        zip_ref.extractall(filename.split('.')[0])
    os.remove(filename+'.zip')
    os.remove(filename+'.fasta')

config_file = args.config

config = process_config(config_file)
print("#####Stage 3#####")
print('Here is the configuration of this run: ')
print(config)

os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu
os.environ['MKL_THREADING_LAYER'] = 'GNU'

d = config.u_net_d
BATCH_SIZE = config.BATCH_SIZE
OUT_STEP = config.OUT_STEP
LOAD_MODEL = config.LOAD_MODEL
pp_steps = config.pp_steps
pp_loss = config.pp_loss
data_type = config.data_type
model_type = config.model_type
pp_type = '{}_s{}'.format(config.pp_model, pp_steps)
rho_per_position = config.rho_per_position
e2e_model_path = '../models_ckpt/e2e_{}_{}_d{}_{}_{}_position_{}.pt'.format(model_type,
    pp_type,d, data_type, pp_loss,rho_per_position)
epoches_third = config.epoches_third
evaluate_epi = config.evaluate_epi
step_gamma = config.step_gamma
k = config.k

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# seed everything for reproduction
seed_torch(0)

seq_len = 500

if model_type =='test_lc':
    contact_net = ContactNetwork_test(d=d, L=seq_len).to(device)
if model_type == 'att6':
    contact_net = ContactAttention(d=d, L=seq_len).to(device)
if model_type == 'att_simple':
    contact_net = ContactAttention_simple(d=d, L=seq_len).to(device)    
if model_type == 'att_simple_fix':
    contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len, 
        device=device).to(device)
if model_type == 'fc':
    contact_net = ContactNetwork_fc(d=d, L=seq_len).to(device)
if model_type == 'conv2d_fc':
    contact_net = ContactNetwork(d=d, L=seq_len).to(device)
if model_type == 'no_noise':
    contact_net = No_Noise(d=d, L=seq_len, device=device).to(device)

# need to write the class for the computational graph of lang pp
if pp_type=='nn':
    lag_pp_net = Lag_PP_NN(pp_steps, k).to(device)
if 'zero' in pp_type:
    lag_pp_net = Lag_PP_zero(pp_steps, k).to(device)
if 'perturb' in pp_type:
    lag_pp_net = Lag_PP_perturb(pp_steps, k).to(device)
if 'mixed'in pp_type:
    lag_pp_net = Lag_PP_mixed(pp_steps, k, rho_per_position).to(device)

rna_ss_e2e = RNA_SS_e2e(contact_net, lag_pp_net)

print('path is: ', e2e_model_path)
if LOAD_MODEL and os.path.isfile(e2e_model_path):
    print('Loading e2e model...')
    rna_ss_e2e.load_state_dict(torch.load(e2e_model_path))


# load test sequences
folder = config.test_folder
files = os.listdir(folder)
# files = list(filter(lambda x: 'fasta' in x, files))

# load extra features
# np_folder = config.np_folder
prob_maps = list()

sequences = list()

length_dict = {}

"""
def load_seq(file_name):
    with open(file_name, 'r') as f:
        t = f.read()
        l = t.splitlines()
    return ''.join(l)
"""

def load_seq(file_name):
    with open(file_name, 'r') as f:
        seq = f.readlines()[-1].strip()
        seq = seq.upper().replace("T", "U").replace(".", "N").replace("_", "N").replace("-", "N")

    #record_iter = SeqIO.parse(file_name, "fasta")
    #for index, record in enumerate(record_iter):
    #    seq = record.seq
    #    seq = seq.upper().replace("T", "U").replace(".", "N").replace("_", "N").replace("-", "N")

    return seq

def load_feat(file_name):
    return np.load(file_name)

def padding2(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0,maxlen-a),(0,maxlen-b)), 'constant')


# original using api
"""
for file_name in files:
    sequences.append(load_seq(os.path.join(folder,file_name)))
    convert_to_fasta(os.path.join(folder,file_name))
    get_rna_ss(os.path.join(folder,file_name))
    path = os.path.join(folder,file_name.split('.')[0])
    fn = glob.glob(path+'/r-ss/*.npy')
    my_arr = load_feat(fn[0])
    prob_maps.append(my_arr)
    shutil.rmtree(path)
"""
# prob_dir = os.path.join("/user/liyu/cjy/RNA/methods/E2Efold-FM/e2efold_productive/ss-rnafm/r-ss")
# prob_dir = os.path.join("/data/tanqingxiong/E2EfoldFM_code/e2efoldFM_productive/ss-rnafm/r-ss")
prob_dir = os.path.join("./ss-rnafm/r-ss")
for file_name in files:
    #print(file_name)
    seq = load_seq(os.path.join(folder,file_name))
    length_dict[file_name] = len(seq)
    sequences.append(seq)
    #print(sequences)
    prob_map_file = os.path.join(prob_dir, file_name.replace(".fasta", ".npy"))
    my_arr = np.load(prob_map_file)
    #print(my_arr)
    #print(my_arr.shape)
    #exit()
    prob_maps.append(my_arr)
print("finish data creation")

querys = list(zip(files, sequences, prob_maps))
querys = list(filter(lambda x: len(x[1])<=seq_len, querys))


# make the predictions

contact_net.eval()
lag_pp_net.eval()

final_result_dict = dict()

names, sequences, prob_maps = zip(*querys)
seq_batch = np.array_split(np.array(sequences), 
    math.ceil(len(sequences)/BATCH_SIZE))
prob_map_batch = np.array_split(np.array(prob_maps),
    math.ceil(len(prob_maps)/BATCH_SIZE))

ct_list = list()
npy_list = list()

for index, (seqs, pr_maps) in enumerate(zip(seq_batch, prob_map_batch)):
    print("{}/{}".format(index, len(seq_batch)))
    seq_embeddings = list(map(seq_encoding, seqs))
    seq_embeddings = list(map(lambda x: padding(x, seq_len), 
        seq_embeddings))
    seq_embeddings = np.array(seq_embeddings)
    seq_lens = torch.Tensor(np.array(list(map(len, seqs)))).int()

    seq_embedding_batch = torch.Tensor(seq_embeddings).float().to(device)

    state_pad = torch.zeros(1,2,2).to(device)

    PE_batch = get_pe(seq_lens, seq_len).float().to(device)
    pr_maps = list(map(lambda x: padding2(x, 500), pr_maps))
    pr_maps = np.array(pr_maps)
    pr_maps = torch.Tensor(pr_maps).float().to(device)

    with torch.no_grad():
        pred_contacts = contact_net(PE_batch, 
            seq_embedding_batch, state_pad, pr_maps)
        a_pred_list = lag_pp_net(pred_contacts, seq_embedding_batch)

    # the learning pp result
    final_pred = (a_pred_list[-1].cpu()>0.5).float()

    for i in range(final_pred.shape[0]):
        ct_tmp = contact2ct(final_pred[i].cpu().numpy(), 
            seq_embeddings[i], seq_lens.numpy()[i])
        ct_list.append(ct_tmp)

        npy_list.append(final_pred[i])

    # for saving the results
save_path = config.save_folder
if not os.path.exists(save_path):
    os.makedirs(save_path)

def save_file(folder, file, ct_contact):
        file_path = os.path.join(folder, file)
        first_line = str(len(ct_contact)) + '\t' + file + '\n'
        content = ct_contact.to_csv(header=None, index=None, sep='\t')
        with open(file_path, 'w') as f:
                f.write(first_line+content)


save_npy_dir = "./npy"
if os.path.exists(save_npy_dir) != True:
    os.makedirs(save_npy_dir)

for i in range(len(names)):
    save_file(save_path, names[i]+'.ct', ct_list[i])
    length = length_dict[names[i]]
    npy_resize = npy_list[i][:length, :length].numpy()
    print(i, npy_resize.shape)
    np.save(os.path.join(save_npy_dir, names[i].replace(".fasta", ".npy")), npy_resize)
    


