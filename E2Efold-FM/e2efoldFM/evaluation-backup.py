import torch
from e2efold.common.utils import *
from e2efold.postprocess import postprocess
import _pickle as pickle

# randomly select one sample from the test set and perform the evaluation
def model_eval(val_generator, contact_net, lag_pp_net, device):
    contact_net.eval()
    lag_pp_net.eval()
    contacts, seq_embeddings, matrix_reps, seq_lens = next(iter(val_generator))
    contacts_batch = torch.Tensor(contacts.float()).to(device)
    seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
    matrix_reps_batch = torch.unsqueeze(
        torch.Tensor(matrix_reps.float()).to(device), -1)

    # padding the states for supervised training with all 0s
    state_pad = torch.zeros(contacts.shape).to(device)
    PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().to(device)
    with torch.no_grad():
        pred_contacts = contact_net(PE_batch, 
            seq_embedding_batch, state_pad)
        a_pred_list = lag_pp_net(pred_contacts, seq_embedding_batch)

    final_pred = (a_pred_list[-1].cpu()>0.5).float()

    result_tuple_list = list(map(lambda i: evaluate_exact(final_pred.cpu()[i], 
        contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
    exact_p,exact_r,exact_f1 = zip(*result_tuple_list)
    print('Average testing precision: ', np.average(exact_p))
    print('Average testing recall score: ', np.average(exact_r))
    print('Average testing f1 score: ', np.average(exact_f1))

    result_tuple_list_shift = list(map(lambda i: evaluate_shifted(final_pred.cpu()[i], 
        contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
    shift_p,shift_r,shift_f1 = zip(*result_tuple_list_shift)  
    print('Average testing precision allow shift: ', np.average(shift_p))
    print('Average testing recall score allow shift: ', np.average(shift_r))
    print('Average testing f1 score allow shift: ', np.average(shift_f1))




def model_eval_all_test(test_generator, contact_net, lag_pp_net, device):
    contact_net.eval()
    lag_pp_net.eval()
    result_no_train = list()
    result_no_train_shift = list()
    result_pp = list()
    result_pp_shift = list()

    f1_no_train = list()
    f1_pp = list()
    seq_lens_list = list()

    batch_n = 0
    for contacts, seq_embeddings, matrix_reps, seq_lens in test_generator:
        if batch_n %10==0:
            print('Batch number: ', batch_n)
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        matrix_reps_batch = torch.unsqueeze(
            torch.Tensor(matrix_reps.float()).to(device), -1)

        state_pad = torch.zeros(contacts.shape).to(device)

        PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().to(device)
        with torch.no_grad():
            pred_contacts = contact_net(PE_batch, 
                seq_embedding_batch, state_pad)
            a_pred_list = lag_pp_net(pred_contacts, seq_embedding_batch)

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

        f1_no_train_tmp = list(map(lambda i: F1_low_tri(map_no_train.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        f1_no_train += f1_no_train_tmp

        # the learning pp result
        final_pred = (a_pred_list[-1].cpu()>0.5).float()
        result_tmp = list(map(lambda i: evaluate_exact(final_pred.cpu()[i], 
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp += result_tmp

        result_tmp_shift = list(map(lambda i: evaluate_shifted(final_pred.cpu()[i], 
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp_shift += result_tmp_shift

        f1_tmp = list(map(lambda i: F1_low_tri(final_pred.cpu()[i], 
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        f1_pp += f1_tmp
        seq_lens_list += list(seq_lens)



    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)
    nt_shift_p,nt_shift_r,nt_shift_f1 = zip(*result_no_train_shift)  

    pp_exact_p,pp_exact_r,pp_exact_f1 = zip(*result_pp)
    pp_shift_p,pp_shift_r,pp_shift_f1 = zip(*result_pp_shift)  
    print('Average testing F1 score with learning post-processing: ', np.average(pp_exact_f1))
    print('Average testing F1 score with zero parameter pp: ', np.average(nt_exact_f1))

    print('Average testing F1 score with learning post-processing allow shift: ', np.average(pp_shift_f1))
    print('Average testing F1 score with zero parameter pp allow shift: ', np.average(nt_shift_f1))

    print('Average testing precision with learning post-processing: ', np.average(pp_exact_p))
    print('Average testing precision with zero parameter pp: ', np.average(nt_exact_p))

    print('Average testing precision with learning post-processing allow shift: ', np.average(pp_shift_p))
    print('Average testing precision with zero parameter pp allow shift: ', np.average(nt_shift_p))

    print('Average testing recall with learning post-processing: ', np.average(pp_exact_r))
    print('Average testing recall with zero parameter pp : ', np.average(nt_exact_r))

    print('Average testing recall with learning post-processing allow shift: ', np.average(pp_shift_r))
    print('Average testing recall with zero parameter pp allow shift: ', np.average(nt_shift_r))

    result_dict = dict()
    result_dict['exact_p'] = pp_exact_p
    result_dict['exact_r'] = pp_exact_r
    result_dict['exact_f1'] = pp_exact_f1
    result_dict['shift_p'] = pp_shift_p
    result_dict['shift_r'] = pp_shift_r
    result_dict['shift_f1'] = pp_shift_f1
    result_dict['seq_lens'] = seq_lens_list
    result_dict['exact_weighted_f1'] = np.sum(np.array(pp_exact_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list))
    result_dict['shift_weighted_f1'] = np.sum(np.array(pp_shift_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list))
    # with open('../results/rnastralign_short_e2e_evaluation_dict.pickle', 'wb') as f:
    #     pickle.dump(result_dict, f)

def all_test_only_e2e(test_generator, contact_net, lag_pp_net, device, test_data):
    contact_net.eval()
    lag_pp_net.eval()
    result_no_train = list()
    result_no_train_shift = list()
    result_pp = list()
    result_pp_shift = list()

    f1_no_train = list()
    f1_pp = list()
    seq_lens_list = list()

    batch_n = 0
    for contacts, seq_embeddings, matrix_reps, seq_lens in test_generator:
        if batch_n %10==0:
            print('Batch number: ', batch_n)
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        matrix_reps_batch = torch.unsqueeze(
            torch.Tensor(matrix_reps.float()).to(device), -1)

        state_pad = torch.zeros(contacts.shape).to(device)

        PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().to(device)
        with torch.no_grad():
            pred_contacts = contact_net(PE_batch, 
                seq_embedding_batch, state_pad)
            a_pred_list = lag_pp_net(pred_contacts, seq_embedding_batch)

        # the learning pp result
        final_pred = (a_pred_list[-1].cpu()>0.5).float()
        result_tmp = list(map(lambda i: evaluate_exact(final_pred.cpu()[i], 
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp += result_tmp

        result_tmp_shift = list(map(lambda i: evaluate_shifted(final_pred.cpu()[i], 
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp_shift += result_tmp_shift

        f1_tmp = list(map(lambda i: F1_low_tri(final_pred.cpu()[i], 
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        f1_pp += f1_tmp
        seq_lens_list += list(seq_lens)


    pp_exact_p,pp_exact_r,pp_exact_f1 = zip(*result_pp)
    pp_shift_p,pp_shift_r,pp_shift_f1 = zip(*result_pp_shift)
    # the following is inserted for arxiv ii
    pp_exact_p = np.nan_to_num(np.array(pp_exact_p))
    pp_exact_r = np.nan_to_num(np.array(pp_exact_r))
    pp_exact_f1 = np.nan_to_num(np.array(pp_exact_f1))

    pp_shift_p = np.nan_to_num(np.array(pp_shift_p))
    pp_shift_r = np.nan_to_num(np.array(pp_shift_r))
    pp_shift_f1 = np.nan_to_num(np.array(pp_shift_f1))
    print('Average testing F1 score with learning post-processing: ', np.average(pp_exact_f1))
    print('Average testing F1 score with learning post-processing allow shift: ', np.average(pp_shift_f1))

    print('Average testing precision with learning post-processing: ', np.average(pp_exact_p))
    print('Average testing precision with learning post-processing allow shift: ', np.average(pp_shift_p))

    print('Average testing recall with learning post-processing: ', np.average(pp_exact_r))
    print('Average testing recall with learning post-processing allow shift: ', np.average(pp_shift_r))

    result_dict = dict()
    result_dict['exact_p'] = pp_exact_p
    result_dict['exact_r'] = pp_exact_r
    result_dict['exact_f1'] = pp_exact_f1
    result_dict['shift_p'] = pp_shift_p
    result_dict['shift_r'] = pp_shift_r
    result_dict['shift_f1'] = pp_shift_f1
    result_dict['seq_lens'] = seq_lens_list
    result_dict['exact_weighted_f1'] = np.sum(np.array(pp_exact_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list))
    result_dict['shift_weighted_f1'] = np.sum(np.array(pp_shift_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list))
    result_dict['name'] = [a.name for a in test_data.data]
    # with open('../results/rnastralign_short_e2e_evaluation_dict.pickle', 'wb') as f:
    #     pickle.dump(result_dict, f)
    # with open('../results/archiveii_short_e2e_evaluation_dict.pickle', 'wb') as f:
    #     pickle.dump(result_dict, f)
