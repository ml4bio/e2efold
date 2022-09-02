import torch
import math
import numpy as np
import torch.nn.functional as F
from e2efold.common.utils import unravel2d_torch, F1_low_tri, encoding2seq
from e2efold.common.utils import constraint_matrix, constraint_matrix_batch

def logit2binary(pred_contacts):
    sigmoid_results = torch.sigmoid(pred_contacts)
    binary = torch.where(sigmoid_results > 0.9, 
        torch.ones(pred_contacts.shape), 
        torch.zeros(pred_contacts.shape))
    return binary


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


def contact_a(a_hat, m):
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a


def aug_lagrangian(u, m, a_hat, lmbd):
    a = contact_a(a_hat, m)
    f = - torch.sum(a * u)/2 + torch.sum(lmbd * F.relu(torch.sum(a, dim=-1) - 1))
    return f


def sign(x):
    return (x > 0).type(x.dtype)


def soft_sign(x):
    k = 1
    return 1.0/(1.0+torch.exp(-2*k*x))


def postprocess(u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False):
    """
    :param u: utility matrix, u is assumed to be symmetric, in batch
    :param x: RNA sequence, in batch
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    m = constraint_matrix_batch(x)
    # u with threshold
    # equivalent to sigmoid(u) > 0.9
    # u = (u > math.log(9.0)).type(torch.FloatTensor) * u
    u = soft_sign(u - math.log(9.0)) * u

    # initialization
    a_hat = (torch.sigmoid(u)) * soft_sign(u - math.log(9.0)).detach()
    lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()

    # gradient descent
    for t in range(num_itr):

        grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        a_hat -= lr_min * grad
        lr_min = lr_min * 0.99

        if with_l1:
            a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)

        lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
        lmbd += lr_max * lmbd_grad
        lr_max = lr_max * 0.99

        # print
        # if t % 20 == 19:
        #     n1 = torch.norm(lmbd_grad)
        #     grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
        #     grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        #     n2 = torch.norm(grad)
        #     print([t, 'norms', n1, n2, aug_lagrangian(u, m, a_hat, lmbd), torch.sum(contact_a(a_hat, u))])

    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a
