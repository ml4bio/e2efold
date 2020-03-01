import os
from e2efold.common.config import process_config
from e2efold.common.utils import get_args
args = get_args()

config_file = args.config

config = process_config(config_file)
print("#####Stage 1#####")
print('Here is the configuration of this run: ')
print(config)
os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu

import torch.optim as optim
from torch.utils import data

from e2efold.models import ContactNetwork, ContactNetwork_test, ContactNetwork_fc
from e2efold.models import ContactAttention, ContactAttention_simple_fix_PE
from e2efold.models import ContactAttention_simple
from e2efold.common.utils import *
from e2efold.common.long_seq_pre_post_process import *
from e2efold.postprocess import postprocess


d = config.u_net_d
BATCH_SIZE = config.batch_size_stage_1
OUT_STEP = config.OUT_STEP
LOAD_MODEL = config.LOAD_MODEL
pp_steps = config.pp_steps
data_type = config.data_type
model_type = config.model_type
model_path = '../models_ckpt/supervised_{}_{}_d{}_l3.pt'.format(model_type, data_type,d)
epoches_first = config.epoches_first
evaluate_epi = config.evaluate_epi_stage_1


steps_done = 0
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_torch()

# for loading data
# loading the rna ss data, the data has been preprocessed
# 5s data is just a demo data, which do not have pseudoknot, will generate another data having that
from e2efold.data_generator import RNASSDataGenerator, Dataset, Dataset_1800
import collections
RNA_SS_data = collections.namedtuple('RNA_SS_data', 
    'seq ss_label length name pairs')

train_data = RNASSDataGenerator('../data/{}/'.format(data_type), 'train_600')
val_data = RNASSDataGenerator('../data/{}/'.format(data_type), 'val_600')
if data_type == 'archiveII_all':
    test_data = RNASSDataGenerator('../data/{}/'.format(data_type), 'test_600')
if data_type == 'rnastralign_all':
    test_data = RNASSDataGenerator('../data/{}/'.format(data_type), 'test_no_redundant_600')

train_data_1800 = RNASSDataGenerator('../data/{}/'.format(data_type), 'train_1800')
val_data_1800 = RNASSDataGenerator('../data/{}/'.format(data_type), 'val_1800')
if data_type == 'archiveII_all':
    test_data_1800 = RNASSDataGenerator('../data/{}/'.format(data_type), 'test_1800')
if data_type == 'rnastralign_all':
    test_data_1800 = RNASSDataGenerator('../data/{}/'.format(data_type), 'test_no_redundant_1800')


seq_len = train_data.data_y.shape[-2]
print('Max seq length ', seq_len)


# using the pytorch interface to parallel the data generation and model training
params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}
train_set = Dataset(train_data)
train_generator = data.DataLoader(train_set, **params)

val_set = Dataset(val_data)
val_generator = data.DataLoader(val_set, **params)

params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': False}
train_set_1800 = Dataset_1800(train_data_1800)
train_generator_1800 = data.DataLoader(train_set_1800, **params)

val_set_1800 = Dataset_1800(val_data_1800)
val_generator_1800 = data.DataLoader(val_set_1800, **params)

params = {'batch_size': BATCH_SIZE,
          'shuffle': False,
          'num_workers': 6,
          'drop_last': True}
test_set = Dataset(test_data)
test_generator = data.DataLoader(test_set, **params)

params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 6,
          'drop_last': False}
test_set_1800 = Dataset_1800(test_data_1800)
test_generator_1800 = data.DataLoader(test_set_1800, **params)


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


if LOAD_MODEL and os.path.isfile(model_path):
    print('Loading u net model...')
    contact_net.load_state_dict(torch.load(model_path))


u_optimizer = optim.Adam(contact_net.parameters())

# for length as 600
pos_weight = torch.Tensor([300]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
    pos_weight = pos_weight)


# randomly select one sample from the test set and perform the evaluation
def model_eval():
    contact_net.eval()
    contacts, seq_embeddings, matrix_reps, seq_lens = next(iter(val_generator))
    contacts_batch = torch.Tensor(contacts.float()).to(device)
    seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

    # padding the states for supervised training with all 0s
    state_pad = torch.zeros(1,2,1).to(device)
    PE_batch = get_pe(seq_lens, 600).float().to(device)

    with torch.no_grad():
        pred_contacts = contact_net(PE_batch, 
            seq_embedding_batch, state_pad)

    u_no_train = postprocess(pred_contacts,
        seq_embedding_batch, 0.01, 0.1, 50, 1.0, True)
    map_no_train = (u_no_train > 0.5).float()
    f1_no_train_tmp = list(map(lambda i: F1_low_tri(map_no_train.cpu()[i],
        contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
    print('Average val F1 score for 600 with pure post-processing: ', np.average(f1_no_train_tmp))

    seq_embedding_batch, PE_batch, contacts_batch, _, _, _, _ = next(iter(val_generator_1800))
    seq_embedding_batch = seq_embedding_batch[0].to(device)
    PE_batch = PE_batch[0].to(device)
    contacts_batch = contacts_batch[0]
    # padding the states for supervised training with all 0s
    state_pad = torch.zeros(1,2,2).to(device)
    with torch.no_grad():
        pred_contacts = contact_net(PE_batch, 
            seq_embedding_batch, state_pad)
    u_no_train = postprocess(pred_contacts,
        seq_embedding_batch, 0.01, 0.1, 50, 1.0, True)
    map_no_train = (u_no_train > 0.5).float()
    f1_no_train_tmp = list(map(lambda i: F1_low_tri(map_no_train.cpu()[i],
        contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
    print('Average val F1 score for 1800 with pure post-processing: ', np.average(f1_no_train_tmp))


def model_eval_all_test():
    contact_net.eval()
    result_no_train = list()
    result_no_train_shift = list()
    seq_lens_list = list()
    batch_n = 0
    # for contacts, seq_embeddings, matrix_reps, seq_lens in test_generator:
    #     if batch_n%10==0:
    #         print('Batch number: ', batch_n)
    #     batch_n += 1
    #     contacts_batch = torch.Tensor(contacts.float()).to(device)
    #     seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

    #     state_pad = torch.zeros(1,2,2).to(device)

    #     PE_batch = get_pe(seq_lens, 600).float().to(device)
    #     with torch.no_grad():
    #         pred_contacts = contact_net(PE_batch, 
    #             seq_embedding_batch, state_pad)

    #     # only post-processing without learning
    #     u_no_train = postprocess(pred_contacts,
    #         seq_embedding_batch, 0.01, 0.1, 50, 1.0, True)
    #     map_no_train = (u_no_train > 0.5).float()
    #     result_no_train_tmp = list(map(lambda i: evaluate_exact(map_no_train.cpu()[i],
    #         contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
    #     result_no_train += result_no_train_tmp
    #     result_no_train_tmp_shift = list(map(lambda i: evaluate_shifted(map_no_train.cpu()[i],
    #         contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
    #     result_no_train_shift += result_no_train_tmp_shift


    for seq_embedding_batch, PE_batch, contacts_batch, _, _, _, seq_lens in test_generator_1800:
        if batch_n%10==0:
            print('Batch number: ', batch_n)
        batch_n += 1
        seq_embedding_batch = seq_embedding_batch[0].to(device)
        PE_batch = PE_batch[0].to(device)
        contacts_batch = contacts_batch[0]
        # padding the states for supervised training with all 0s
        state_pad = torch.zeros(1,2,2).to(device)

        with torch.no_grad():
            pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
            seq_embedding_batch, 0.01, 0.1, 50, 1.0, True)
        map_no_train = (u_no_train > 0.5).float()
        result_no_train_tmp = list(map(lambda i: evaluate_exact(map_no_train.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp
        result_no_train_tmp_shift = list(map(lambda i: evaluate_shifted(map_no_train.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train_shift += result_no_train_tmp_shift
        seq_lens_list += list(seq_lens)


    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)
    nt_shift_p,nt_shift_r,nt_shift_f1 = zip(*result_no_train_shift)

    nt_exact_p = np.nan_to_num(np.array(nt_exact_p))
    nt_exact_r = np.nan_to_num(np.array(nt_exact_r))
    nt_exact_f1 = np.nan_to_num(np.array(nt_exact_f1))

    nt_shift_p = np.nan_to_num(np.array(nt_shift_p))
    nt_shift_r = np.nan_to_num(np.array(nt_shift_r))
    nt_shift_f1 = np.nan_to_num(np.array(nt_shift_f1))
    
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
    print('Average testing F1 score with pure post-processing allow shift: ', np.average(nt_shift_f1))
    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing precision with pure post-processing allow shift: ', np.average(nt_shift_p))
    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))
    print('Average testing recall with pure post-processing allow shift: ', np.average(nt_shift_r))
    nt_exact_f1_agg = list()
    nt_shift_f1_agg = list()
    for i in range(len(seq_lens_list)):
        nt_exact_f1_agg.append(np.average(nt_exact_f1[i*15:(i+1)*15]))
        nt_shift_f1_agg.append(np.average(nt_shift_f1[i*15:(i+1)*15]))
    result_dict = dict()
    result_dict['exact_p'] = nt_exact_p
    result_dict['exact_r'] = nt_exact_r
    result_dict['exact_f1'] = nt_exact_f1
    result_dict['shift_p'] = nt_shift_p
    result_dict['shift_r'] = nt_shift_r
    result_dict['shift_f1'] = nt_shift_f1
    result_dict['seq_lens'] = seq_lens_list
    result_dict['exact_weighted_f1'] = np.sum(np.array(nt_exact_f1_agg)*np.array(seq_lens_list)/np.sum(seq_lens_list))
    result_dict['shift_weighted_f1'] = np.sum(np.array(nt_shift_f1_agg)*np.array(seq_lens_list)/np.sum(seq_lens_list))
    import _pickle as pickle
    with open('../results/rnastralign_long_pure_pp_evaluation_dict.pickle', 'wb') as f:
        pickle.dump(result_dict, f)

def model_eval_all_test_greedy_sort():
    contact_net.eval()
    result_no_train = list()
    result_no_train_shift = list()
    seq_lens_list = list()
    batch_n = 0
    for seq_embedding_batch, PE_batch, contacts_batch, comb_index, _, contacts, seq_lens in test_generator_1800:
        if batch_n%10==0:
            print('Batch number: ', batch_n)
        batch_n += 1
        seq_embedding_batch = seq_embedding_batch[0].to(device)
        PE_batch = PE_batch[0].to(device)
        contacts_batch = contacts_batch[0]
        # padding the states for supervised training with all 0s
        state_pad = torch.zeros(1,2,2).to(device)

        with torch.no_grad():
            pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)
            pred_u_map = combine_chunk_u_maps_no_replace(pred_contacts, comb_index, 6)
            pred_u_map = pred_u_map.unsqueeze(0)

        # only post-processing without learning
        map_no_train = conflict_sort(pred_u_map)
        result_no_train_tmp = list(map(lambda i: evaluate_exact(map_no_train[i],
            contacts.float().cpu()[i]), range(contacts.shape[0])))
        result_no_train += result_no_train_tmp
        result_no_train_tmp_shift = list(map(lambda i: evaluate_shifted(map_no_train[i],
            contacts.float().cpu()[i]), range(contacts.shape[0])))
        result_no_train_shift += result_no_train_tmp_shift
        seq_lens_list += list(seq_lens)


    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)
    nt_shift_p,nt_shift_r,nt_shift_f1 = zip(*result_no_train_shift)

    nt_exact_p = np.nan_to_num(np.array(nt_exact_p))
    nt_exact_r = np.nan_to_num(np.array(nt_exact_r))
    nt_exact_f1 = np.nan_to_num(np.array(nt_exact_f1))

    nt_shift_p = np.nan_to_num(np.array(nt_shift_p))
    nt_shift_r = np.nan_to_num(np.array(nt_shift_r))
    nt_shift_f1 = np.nan_to_num(np.array(nt_shift_f1))
    
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
    print('Average testing F1 score with pure post-processing allow shift: ', np.average(nt_shift_f1))
    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing precision with pure post-processing allow shift: ', np.average(nt_shift_p))
    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))
    print('Average testing recall with pure post-processing allow shift: ', np.average(nt_shift_r))
    result_dict = dict()
    result_dict['exact_p'] = nt_exact_p
    result_dict['exact_r'] = nt_exact_r
    result_dict['exact_f1'] = nt_exact_f1
    result_dict['shift_p'] = nt_shift_p
    result_dict['shift_r'] = nt_shift_r
    result_dict['shift_f1'] = nt_shift_f1
    result_dict['seq_lens'] = seq_lens_list
    result_dict['exact_weighted_f1'] = np.sum(np.array(nt_exact_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list))
    result_dict['shift_weighted_f1'] = np.sum(np.array(nt_shift_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list))
    import _pickle as pickle
    with open('../results/rnastralign_long_greedy_sort_evaluation_dict.pickle', 'wb') as f:
        pickle.dump(result_dict, f)

def model_eval_all_test_greedy_sampling():
    contact_net.eval()
    result_no_train = list()
    result_no_train_shift = list()
    seq_lens_list = list()
    batch_n = 0
    for seq_embedding_batch, PE_batch, contacts_batch, comb_index, _, contacts, seq_lens in test_generator_1800:
        if batch_n%10==0:
            print('Batch number: ', batch_n)
        batch_n += 1
        seq_embedding_batch = seq_embedding_batch[0].to(device)
        PE_batch = PE_batch[0].to(device)
        contacts_batch = contacts_batch[0]
        # padding the states for supervised training with all 0s
        state_pad = torch.zeros(1,2,2).to(device)

        with torch.no_grad():
            pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)
            pred_u_map = combine_chunk_u_maps_no_replace(pred_contacts, comb_index, 6)
            pred_u_map = pred_u_map.unsqueeze(0)

        # only post-processing without learning
        map_no_train = conflict_sampling(pred_u_map)
        result_no_train_tmp = list(map(lambda i: evaluate_exact(map_no_train[i],
            contacts.float().cpu()[i]), range(contacts.shape[0])))
        result_no_train += result_no_train_tmp
        result_no_train_tmp_shift = list(map(lambda i: evaluate_shifted(map_no_train[i],
            contacts.float().cpu()[i]), range(contacts.shape[0])))
        result_no_train_shift += result_no_train_tmp_shift
        seq_lens_list += list(seq_lens)


    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)
    nt_shift_p,nt_shift_r,nt_shift_f1 = zip(*result_no_train_shift)

    nt_exact_p = np.nan_to_num(np.array(nt_exact_p))
    nt_exact_r = np.nan_to_num(np.array(nt_exact_r))
    nt_exact_f1 = np.nan_to_num(np.array(nt_exact_f1))

    nt_shift_p = np.nan_to_num(np.array(nt_shift_p))
    nt_shift_r = np.nan_to_num(np.array(nt_shift_r))
    nt_shift_f1 = np.nan_to_num(np.array(nt_shift_f1))
    
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
    print('Average testing F1 score with pure post-processing allow shift: ', np.average(nt_shift_f1))
    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing precision with pure post-processing allow shift: ', np.average(nt_shift_p))
    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))
    print('Average testing recall with pure post-processing allow shift: ', np.average(nt_shift_r))
    result_dict = dict()
    result_dict['exact_p'] = nt_exact_p
    result_dict['exact_r'] = nt_exact_r
    result_dict['exact_f1'] = nt_exact_f1
    result_dict['shift_p'] = nt_shift_p
    result_dict['shift_r'] = nt_shift_r
    result_dict['shift_f1'] = nt_shift_f1
    result_dict['seq_lens'] = seq_lens_list
    result_dict['exact_weighted_f1'] = np.sum(np.array(nt_exact_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list))
    result_dict['shift_weighted_f1'] = np.sum(np.array(nt_shift_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list))
    import _pickle as pickle
    with open('../results/rnastralign_long_greedy_sampling_evaluation_dict.pickle', 'wb') as f:
        pickle.dump(result_dict, f)


# There are three steps of training
# step one: train the u net
for epoch in range(epoches_first):
    contact_net.train()
    print('On short sequence phase:')
    for contacts, seq_embeddings, matrix_reps, seq_lens in train_generator:
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

        # padding the states for supervised training with all 0s
        state_pad = torch.zeros(1,2,2).to(device)

        PE_batch = get_pe(seq_lens, 600).float().to(device)
        contact_masks = torch.Tensor(contact_map_masks(seq_lens, 600)).to(device)
        pred_contacts = contact_net(PE_batch, 
            seq_embedding_batch, state_pad)

        # Compute loss
        loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)

        # print(steps_done)
        if steps_done % OUT_STEP ==0:
            print('Stage 1, epoch for 600: {}, step: {}, loss: {}'.format(
                epoch, steps_done, loss_u))
            u_no_train = postprocess(pred_contacts,
                seq_embedding_batch, 0.01, 0.1, 50, 1.0, True)
            map_no_train = (u_no_train > 0.5).float()
            f1_no_train_tmp = list(map(lambda i: F1_low_tri(map_no_train.cpu()[i],
                contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
            print('Average train F1 score for 600 with pure post-processing: ', np.average(f1_no_train_tmp))

        # Optimize the model
        u_optimizer.zero_grad()
        loss_u.backward()
        u_optimizer.step()
        steps_done=steps_done+1
        if steps_done % 600 ==0:
            break

    print('On long sequence phase:')
    u_optimizer.zero_grad()
    for seq_embedding_batch, PE_batch, contacts_batch, _, _, _, _ in train_generator_1800:

        # padding the states for supervised training with all 0s
        state_pad = torch.zeros(1,2,2).to(device)
        seq_embedding_batch = seq_embedding_batch[0].to(device)
        PE_batch = PE_batch[0].to(device)
        contacts_batch = contacts_batch[0].to(device)

        pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)

        # Compute loss
        loss_u = criterion_bce_weighted(pred_contacts, contacts_batch)

        # print(steps_done)
        if steps_done % OUT_STEP ==0:
            print('Stage 1, epoch for 1800: {},step: {}, loss: {}'.format(
                epoch, steps_done, loss_u))
            u_no_train = postprocess(pred_contacts,
                seq_embedding_batch, 0.01, 0.1, 50, 1.0, True)
            map_no_train = (u_no_train > 0.5).float()
            f1_no_train_tmp = list(map(lambda i: F1_low_tri(map_no_train.cpu()[i],
                contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
            print('Average train F1 score for 1800 with pure post-processing: ', np.average(f1_no_train_tmp))

        # Optimize the model
        loss_u.backward()
        if steps_done % 5 ==0:
            u_optimizer.step()
            u_optimizer.zero_grad()
        steps_done=steps_done+1
        if steps_done % 150 ==0:
            break

    if epoch%evaluate_epi==0:
        model_eval()
        torch.save(contact_net.state_dict(), model_path)

# model_eval_all_test()

# sys.exit()







