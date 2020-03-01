#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math
from e2efold.common.utils import soft_sign
import numpy as np
from scipy.sparse import diags


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride=1, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.pad = int((kernel_size-1)/2)
        self.stride = _pair(stride) 
        
    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out



class ResNetblock(nn.Module):

    def __init__(self, conv, in_planes, planes, kernel_size=9, padding=8, dilation=2):
        super(ResNetblock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.bn1_2 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv(in_planes, planes, 
            kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(planes)
        self.bn2_2 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, 
            kernel_size=kernel_size, padding=padding, dilation=dilation)


    def forward(self, x):
        residual = x

        if len(x.shape) == 3:
            out = self.bn1(x)
        else:
            out = self.bn1_2(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        if len(out.shape) ==3:
            out = self.bn2(out)
        else:
            out = self.bn2_2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out

class ContactAttention(nn.Module):
    """
    The definition of contact network
    Instantiation: 
        d: the dimension of the hidden dimension of each base
        L: sequence length
    Input: The sequence encoding, the prior knowledge
    Output: The contact map of the input RNA sequence
    """
    def __init__(self, d, L):
        super(ContactAttention, self).__init__()
        self.d = d
        self.L = L
        # 1d convolution, L*3 to L*d
        self.conv1d1= nn.Conv1d(in_channels=4, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn1 = nn.BatchNorm1d(d)
        self.conv1d2= nn.Conv1d(in_channels=d, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn2 = nn.BatchNorm1d(d)

        self.position_embedding_1d = nn.Parameter(
            torch.randn(1, d, L)
        )

        # transformer encoder for the input sequences
        self.encoder_layer = nn.TransformerEncoderLayer(2*d, 2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 3)
        
        self.lc = LocallyConnected2d(4*d, 1, L, 1)
    def forward(self, prior, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))
        seq = F.relu(self.bn2(self.conv1d2(seq))) # d*L
        position_embeds = self.position_embedding_1d.repeat(seq.shape[0],1,1)

        seq = torch.cat([seq, position_embeds], 1)
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))
        seq = seq.permute(1, 2, 0)

        # what about apply attention on the the 2d map?
        seq_mat = self.matrix_rep(seq) # 4d*L*L

        infor = seq_mat

        contact = self.lc(infor)
        contact = contact.view(-1, self.L, self.L)
        contact = (contact+torch.transpose(contact, -1, -2))/2

        return contact.view(-1, self.L, self.L)
    def matrix_rep(self, x):
        '''
        for each position i,j of the matrix, we concatenate the embedding of i and j
        '''
        x = x.permute(0, 2, 1) # L*d
        L = x.shape[1]
        x2 = x
        x = x.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        x = x.repeat(1, L,1,1)
        x2 = x2.repeat(1, 1, L,1)
        mat = torch.cat([x,x2],-1) # L*L*2d

        # make it symmetric
        # mat_tril = torch.cat(
        #     [torch.tril(mat[:,:, i]) for i in range(mat.shape[-1])], -1)
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2)) # 2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag
        return mat

class ContactAttention_simple(nn.Module):
    """docstring for ContactAttention_simple"""
    def __init__(self, d,L):
        super(ContactAttention_simple, self).__init__()
        self.d = d
        self.L = L
        self.conv1d1= nn.Conv1d(in_channels=4, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn1 = nn.BatchNorm1d(d)

        self.conv_test_1 = nn.Conv2d(in_channels=6*d, out_channels=d, kernel_size=1)
        self.bn_conv_1 = nn.BatchNorm2d(d)
        self.conv_test_2 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1)
        self.bn_conv_2 = nn.BatchNorm2d(d)
        self.conv_test_3 = nn.Conv2d(in_channels=d, out_channels=1, kernel_size=1)

        self.position_embedding_1d = nn.Parameter(
            torch.randn(1, d, 600)
        )

        # transformer encoder for the input sequences
        self.encoder_layer = nn.TransformerEncoderLayer(2*d, 2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 3)

    def forward(self, prior, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """

        position_embeds = self.position_embedding_1d.repeat(seq.shape[0],1,1)
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq))) #d*L just for increase the capacity

        seq = torch.cat([seq, position_embeds], 1) # 2d*L
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))
        seq = seq.permute(1, 2, 0)

        # what about apply attention on the the 2d map?
        seq_mat = self.matrix_rep(seq) # 4d*L*L
        
        p_mat = self.matrix_rep(position_embeds) # 2d*L*L

        infor = torch.cat([seq_mat, p_mat], 1) # 6d*L*L

        contact = F.relu(self.bn_conv_1(self.conv_test_1(infor)))
        contact = F.relu(self.bn_conv_2(self.conv_test_2(contact)))
        contact = self.conv_test_3(contact)

        contact = contact.view(-1, self.L, self.L)
        contact = (contact+torch.transpose(contact, -1, -2))/2

        return contact.view(-1, self.L, self.L)

    def matrix_rep(self, x):
        '''
        for each position i,j of the matrix, we concatenate the embedding of i and j
        '''
        x = x.permute(0, 2, 1) # L*d
        L = x.shape[1]
        x2 = x
        x = x.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        x = x.repeat(1, L,1,1)
        x2 = x2.repeat(1, 1, L,1)
        mat = torch.cat([x,x2],-1) # L*L*2d

        # make it symmetric
        # mat_tril = torch.cat(
        #     [torch.tril(mat[:,:, i]) for i in range(mat.shape[-1])], -1)
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2)) # 2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag
        return mat


class ContactAttention_simple_fix_PE(ContactAttention_simple):
    """docstring for ContactAttention_simple_fix_PE"""
    def __init__(self, d, L, device):
        super(ContactAttention_simple_fix_PE, self).__init__(d, L)
        self.PE_net = nn.Sequential(
            nn.Linear(111,5*d),
            nn.ReLU(),
            nn.Linear(5*d,5*d),
            nn.ReLU(),
            nn.Linear(5*d,d))

        
    def forward(self, pe, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """
        position_embeds = self.PE_net(pe.view(-1, 111)).view(-1, self.L, self.d) # N*L*111 -> N*L*d
        position_embeds = position_embeds.permute(0, 2, 1) # N*d*L
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq))) #d*L just for increase the capacity

        seq = torch.cat([seq, position_embeds], 1) # 2d*L
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))
        seq = seq.permute(1, 2, 0)

        # what about apply attention on the the 2d map?
        seq_mat = self.matrix_rep(seq) # 4d*L*L
        
        p_mat = self.matrix_rep(position_embeds) # 2d*L*L

        infor = torch.cat([seq_mat, p_mat], 1) # 6d*L*L

        contact = F.relu(self.bn_conv_1(self.conv_test_1(infor)))
        contact = F.relu(self.bn_conv_2(self.conv_test_2(contact)))
        contact = self.conv_test_3(contact)

        contact = contact.view(-1, self.L, self.L)
        contact = (contact+torch.transpose(contact, -1, -2))/2

        return contact.view(-1, self.L, self.L)
        


class ContactAttention_fix_em(nn.Module):
    """
    The definition of contact network
    Instantiation: 
        d: the dimension of the hidden dimension of each base
        L: sequence length
    Input: The sequence encoding, the prior knowledge
    Output: The contact map of the input RNA sequence
    """
    def __init__(self, d, L):
        super(ContactAttention_fix_em, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d = d
        self.L = L
        # 1d convolution, L*3 to L*d
        self.conv1d1= nn.Conv1d(in_channels=4, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn1 = nn.BatchNorm1d(d)
        self.conv1d2= nn.Conv1d(in_channels=d, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn2 = nn.BatchNorm1d(d)

        self.fix_pos_em_1d = torch.Tensor(np.arange(1,L+1)/np.float(L)).view(1,1,L).to(
            self.device)

        pos_j, pos_i = np.meshgrid(np.arange(1,L+1)/np.float(L), 
            np.arange(1,L+1)/np.float(L))
        self.fix_pos_em_2d = torch.cat([torch.Tensor(pos_i).unsqueeze(0), 
            torch.Tensor(pos_j).unsqueeze(0)], 0).unsqueeze(0).to(self.device)

        # transformer encoder for the input sequences
        self.encoder_layer = nn.TransformerEncoderLayer(d+1, 2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 3)

        self.lc = LocallyConnected2d(2*d+2+2, 1, L, 1)
        self.conv_test = nn.Conv2d(in_channels=2*d+2+2, out_channels=1, 
            kernel_size=1)

    def forward(self, prior, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))
        seq = F.relu(self.bn2(self.conv1d2(seq))) # d*L
        position_embeds = self.fix_pos_em_1d.repeat(seq.shape[0],1,1)

        seq = torch.cat([seq, position_embeds], 1)
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))
        seq = seq.permute(1, 2, 0)

        # what about apply attention on the the 2d map?
        seq_mat = self.matrix_rep(seq) # (2d+2)*L*L

        position_embeds_2d = self.fix_pos_em_2d.repeat(seq.shape[0],1,1,1)
        infor = torch.cat([seq_mat, position_embeds_2d], 1) #(2d+2+2)*L*L

        contact = self.lc(infor)
        # contact = self.conv_test(infor)
        contact = contact.view(-1, self.L, self.L)
        contact = (contact+torch.transpose(contact, -1, -2))/2

        return contact.view(-1, self.L, self.L)



    def matrix_rep(self, x):
        '''
        for each position i,j of the matrix, we concatenate the embedding of i and j
        '''
        x = x.permute(0, 2, 1) # L*d
        L = x.shape[1]
        x2 = x
        x = x.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        x = x.repeat(1, L,1,1)
        x2 = x2.repeat(1, 1, L,1)
        mat = torch.cat([x,x2],-1) # L*L*2d

        # make it symmetric
        # mat_tril = torch.cat(
        #     [torch.tril(mat[:,:, i]) for i in range(mat.shape[-1])], -1)
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2)) # 2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag

        return mat


class Lag_PP_NN(nn.Module):
    """
    The definition of Lagrangian post-procssing with neural network parameterization
    Instantiation: 
    :steps: the number of unroll steps
    Input: 
    :u: the utility matrix, batch*L*L
    :s: the sequence encoding, batch*L*4

    Output: a list of contact map of each step, batch*L*L
    """
    def __init__(self, steps, k):
        super(Lag_PP_NN, self).__init__()
        self.steps = steps
        # the parameter for the soft sign
        # the k value need to be tuned
        self.k = k
        self.s = math.log(9.0)
        self.w = 1
        self.rho = 1
        # self.s = nn.Parameter(torch.randn(1))
        # self.w = nn.Parameter(torch.randn(1))
        # self.a_hat_conv_list = nn.ModuleList()
        # self.rho_conv_list = nn.ModuleList()
        # self.lmbd_conv_list = nn.ModuleList()
        # self.make_update_cnns(steps)

        self.a_hat_fc_list = nn.ModuleList()
        self.rho_fc_list = nn.ModuleList()
        self.lmbd_fc_list = nn.ModuleList()        
        self.make_update_fcs(steps)

    def make_update_fcs(self, steps):
        for i in range(steps):
            a_hat_fc_tmp = nn.Sequential(
                nn.Linear(3,3),
                nn.ReLU(),
                nn.Linear(3,1),
                nn.ReLU())
            rho_fc_tmp = nn.Sequential(
                nn.Linear(3,3),
                nn.ReLU(),
                nn.Linear(3,1),
                nn.ReLU())
            lmbd_fc_tmp = nn.Sequential(
                nn.Linear(2,3),
                nn.ReLU(),
                nn.Linear(3,1),
                nn.ReLU())
            self.a_hat_fc_list.append(a_hat_fc_tmp)
            self.rho_fc_list.append(rho_fc_tmp)
            self.lmbd_fc_list.append(lmbd_fc_tmp)

    def make_update_cnns(self, steps):
        for i in range(steps):
            a_hat_conv_tmp = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1),
                nn.ReLU())
            rho_conv_tmp = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1),
                nn.ReLU())
            lmbd_conv_tmp = nn.Sequential(
                nn.Conv1d(in_channels=2, out_channels=3, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1),
                nn.ReLU())
            self.a_hat_conv_list.append(a_hat_conv_tmp)
            self.rho_conv_list.append(rho_conv_tmp)
            self.lmbd_conv_list.append(lmbd_conv_tmp)

    def forward(self, u, x):
        a_t_list = list()
        a_hat_t_list = list()
        lmbd_t_list = list()

        m = self.constraint_matrix_batch(x) # N*L*L

        u = soft_sign(u - self.s, self.k) * u

        # initialization
        a_hat_tmp = (torch.sigmoid(u)) * soft_sign(u - self.s, self.k)
        a_tmp = self.contact_a(a_hat_tmp, m)
        lmbd_tmp = self.w*F.relu(torch.sum(a_tmp, dim=-1) - 1)

        lmbd_t_list.append(lmbd_tmp)
        a_t_list.append(a_tmp)
        a_hat_t_list.append(a_hat_tmp)
        # gradient descent
        for t in range(self.steps):
            lmbd_updated, a_updated, a_hat_updated = self.update_rule_fc(
                u, m, lmbd_tmp, a_tmp, a_hat_tmp, t)

            a_hat_tmp = a_hat_updated
            a_tmp = a_updated
            lmbd_tmp = lmbd_updated

            lmbd_t_list.append(lmbd_tmp)
            a_t_list.append(a_tmp)
            a_hat_t_list.append(a_hat_tmp)

        # return a_updated
        return a_t_list[1:]

    def update_rule_fc(self, u, m, lmbd, a, a_hat, t):

        grad_a = - u / 2 + (lmbd * soft_sign(torch.sum(a,
            dim=-1) - 1, self.k)).unsqueeze_(-1).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        # grad: n*L*L

        # reshape them first: N*L*L*3 => NLL*3
        a_hat_fc = self.a_hat_fc_list[t]
        rho_fc = self.rho_fc_list[t]
        input_features = torch.cat([torch.unsqueeze(a_hat,-1),
            torch.unsqueeze(grad,-1), torch.unsqueeze(u,-1)], -1).view(-1, 3)
        a_hat_updated = a_hat_fc(input_features).view(a_hat.shape)

        rho = rho_fc(input_features).view(a_hat.shape)
        a_hat_updated = F.relu(torch.abs(a_hat_updated) - rho)
        # a_hat_updated = F.relu(torch.abs(a_hat_updated) - self.rho)
        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        # lmbd: n*L, so we should use 1d conv
        lmbd_fc = self.lmbd_fc_list[t]
        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_input_features = torch.cat([torch.unsqueeze(lmbd, -1),
            torch.unsqueeze(lmbd_grad, -1)], -1).view(-1, 2)
        lmbd_updated = lmbd_fc(lmbd_input_features).view(lmbd.shape)

        return lmbd_updated, a_updated, a_hat_updated




    def update_rule(self, u, m, lmbd, a, a_hat, t):

        grad_a = - u / 2 + (lmbd * soft_sign(torch.sum(a,
            dim=-1) - 1, self.k)).unsqueeze_(-1).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        # grad: n*L*L

        # we update the a_hat with 2 conv layers whose filters are 1 by 1
        # so that different positions can share parameters
        # we put a_hat, g and u as three channels and the output a_hat as one channel
        # the inputs are N*3*L*L
        a_hat_conv = self.a_hat_conv_list[t]
        rho_conv = self.rho_conv_list[t]
        input_features = torch.cat([torch.unsqueeze(a_hat,1),
            torch.unsqueeze(grad,1), torch.unsqueeze(u,1)], 1)
        a_hat_updated = torch.squeeze(a_hat_conv(input_features), 1)
        # rho = torch.squeeze(rho_conv(input_features),1)
        # a_hat_updated = F.relu(torch.abs(a_hat) - rho)
        a_hat_updated = F.relu(torch.abs(a_hat_updated) - self.rho)
        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        # lmbd: n*L, so we should use 1d conv
        lmbd_conv = self.lmbd_conv_list[t]
        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_input_features = torch.cat([torch.unsqueeze(lmbd,1),
            torch.unsqueeze(lmbd_grad,1)], 1)
        lmbd_updated = torch.squeeze(lmbd_conv(lmbd_input_features), 1)

        return lmbd_updated, a_updated, a_hat_updated

    def constraint_matrix_batch(self, x):
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
    
    def contact_a(self, a_hat, m):
        a = a_hat * a_hat
        a = (a + torch.transpose(a, -1, -2)) / 2
        a = a * m
        return a


class Lag_PP_zero(nn.Module):
    """
    The definition of Lagrangian post-procssing with no parameters
    Instantiation: 
    :steps: the number of unroll steps
    Input: 
    :u: the utility matrix, batch*L*L
    :s: the sequence encoding, batch*L*4

    Output: a list of contact map of each step, batch*L*L
    """
    def __init__(self, steps, k):
        super(Lag_PP_zero, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = steps
        # the parameter for the soft sign
        self.k = k
        self.s = math.log(9.0)
        self.rho = 1.0
        self.alpha = 0.01
        self.beta = 0.1
        self.lr_decay = 0.99

    def forward(self, u, x):
        a_t_list = list()
        a_hat_t_list = list()
        lmbd_t_list = list()

        m = self.constraint_matrix_batch(x) # N*L*L

        u = soft_sign(u - self.s, self.k) * u

        # initialization
        a_hat_tmp = (torch.sigmoid(u)) * soft_sign(u - self.s, self.k).detach()
        a_tmp = self.contact_a(a_hat_tmp, m)
        lmbd_tmp = F.relu(torch.sum(a_tmp, dim=-1) - 1).detach()

        lmbd_t_list.append(lmbd_tmp)
        a_t_list.append(a_tmp)
        a_hat_t_list.append(a_hat_tmp)
        # gradient descent
        for t in range(self.steps):
            lmbd_updated, a_updated, a_hat_updated = self.update_rule(
                u, m, lmbd_tmp, a_tmp, a_hat_tmp, t)

            a_hat_tmp = a_hat_updated
            a_tmp = a_updated
            lmbd_tmp = lmbd_updated

            lmbd_t_list.append(lmbd_tmp)
            a_t_list.append(a_tmp)
            a_hat_t_list.append(a_hat_tmp)

        # return a_updated
        return a_t_list[1:]

    def update_rule(self, u, m, lmbd, a, a_hat, t):

        grad_a = - u / 2 + (lmbd * soft_sign(torch.sum(a,
            dim=-1) - 1, self.k)).unsqueeze_(-1).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))

        a_hat_updated = a_hat - self.alpha * grad
        self.alpha *= self.lr_decay
        a_hat_updated = F.relu(torch.abs(a_hat_updated) - self.rho*self.alpha)
        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_updated = lmbd + self.beta * lmbd_grad
        self.beta *= self.lr_decay
        
        return lmbd_updated, a_updated, a_hat_updated

    def constraint_matrix_batch(self, x):
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

        mask = diags([1]*7, [-3, -2, -1, 0, 1, 2, 3], 
            shape=(m.shape[-2], m.shape[-1])).toarray()
        m = m.masked_fill(torch.Tensor(mask).bool().to(self.device), 0)
        return m
    
    def contact_a(self, a_hat, m):
        a = a_hat * a_hat
        a = (a + torch.transpose(a, -1, -2)) / 2
        a = a * m
        return a


class Lag_PP_perturb(Lag_PP_zero):
    def __init__(self, steps, k):
        super(Lag_PP_perturb, self).__init__(steps, k)
        self.steps = steps
        self.k = k
        self.lr_decay = nn.Parameter(torch.Tensor([0.99]))
        # self.s = nn.Parameter(torch.Tensor([math.log(9.0)]))
        self.s = math.log(9.0)
        self.rho = nn.ParameterList([nn.Parameter(torch.Tensor([1.0])) for i in range(steps)])
        self.alpha = nn.ParameterList([nn.Parameter(torch.Tensor([0.01*math.pow(self.lr_decay, 
            i)])) for i in range(steps)])
        self.beta = nn.ParameterList([nn.Parameter(torch.Tensor([0.1*math.pow(self.lr_decay, 
            i)])) for i in range(steps)])
        

    def forward(self, u, x):
        a_t_list = list()
        a_hat_t_list = list()
        lmbd_t_list = list()

        m = self.constraint_matrix_batch(x) # N*L*L

        u = soft_sign(u - self.s, self.k) * u

        # initialization
        a_hat_tmp = (torch.sigmoid(u)) * soft_sign(u - self.s, self.k).detach()
        a_tmp = self.contact_a(a_hat_tmp, m)
        lmbd_tmp = F.relu(torch.sum(a_tmp, dim=-1) - 1).detach()

        lmbd_t_list.append(lmbd_tmp)
        a_t_list.append(a_tmp)
        a_hat_t_list.append(a_hat_tmp)
        # gradient descent
        for t in range(self.steps):
            lmbd_updated, a_updated, a_hat_updated = self.update_rule(
                u, m, lmbd_tmp, a_tmp, a_hat_tmp, t)

            a_hat_tmp = a_hat_updated
            a_tmp = a_updated
            lmbd_tmp = lmbd_updated

            lmbd_t_list.append(lmbd_tmp)
            a_t_list.append(a_tmp)
            a_hat_t_list.append(a_hat_tmp)

        # return a_updated
        return a_t_list[1:]

    def update_rule(self, u, m, lmbd, a, a_hat, t):

        grad_a = - u / 2 + (lmbd * soft_sign(torch.sum(a,
            dim=-1) - 1, self.k)).unsqueeze_(-1).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))

        a_hat_updated = a_hat - self.alpha[t] * grad
        a_hat_updated = F.relu(torch.abs(a_hat_updated) - self.rho[t]*self.alpha[t])
        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_updated = lmbd + self.beta[t] * lmbd_grad
        
        return lmbd_updated, a_updated, a_hat_updated


class Lag_PP_mixed(Lag_PP_zero):
    """
    For the update of a and lambda, we use gradient descent with 
    learnable parameters. For the rho, we use neural network to learn
    a position related threshold
    """
    def __init__(self, steps, k, rho_mode='fix'):
        super(Lag_PP_mixed, self).__init__(steps, k)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = steps
        self.k = k
        # self.s = nn.Parameter(torch.ones(600, 600)*math.log(9.0))
        self.s = nn.Parameter(torch.Tensor([math.log(9.0)]))
        # self.s = math.log(9.0)
        self.w = nn.Parameter(torch.randn(1))
        self.rho = nn.Parameter(torch.Tensor([1.0]))
        # self.rho = 1.0
        self.rho_m = nn.Parameter(torch.randn(600, 600))
        self.rho_net = nn.Sequential(
                nn.Linear(3,5),
                nn.ReLU(),
                nn.Linear(5,1),
                nn.ReLU())
        # build the rho network
        # reuse it under every time step
        self.alpha = nn.Parameter(torch.Tensor([0.01]))
        self.beta = nn.Parameter(torch.Tensor([0.1]))
        self.lr_decay_alpha = nn.Parameter(torch.Tensor([0.99]))
        self.lr_decay_beta = nn.Parameter(torch.Tensor([0.99]))
        # self.alpha = torch.Tensor([0.01]).cuda()
        # self.beta = torch.Tensor([0.1]).cuda()
        # self.lr_decay_alpha = torch.Tensor([0.99]).cuda()
        # self.lr_decay_beta = torch.Tensor([0.99]).cuda()
        self.rho_mode = rho_mode

        pos_j, pos_i = np.meshgrid(np.arange(1,600+1)/600.0, 
            np.arange(1,600+1)/600.0)
        self.rho_pos_fea = torch.cat([torch.Tensor(pos_i).unsqueeze(-1), 
            torch.Tensor(pos_j).unsqueeze(-1)], -1).view(-1, 2).to(self.device)

        self.rho_pos_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.ReLU()
            )

    def forward(self, u, x):
        a_t_list = list()
        a_hat_t_list = list()
        lmbd_t_list = list()

        m = self.constraint_matrix_batch(x) # N*L*L

        u = soft_sign(u - self.s, self.k) * u

        # initialization
        a_hat_tmp = (torch.sigmoid(u)) * soft_sign(u - self.s, self.k).detach()
        a_tmp = self.contact_a(a_hat_tmp, m)
        lmbd_tmp = self.w * F.relu(torch.sum(a_tmp, dim=-1) - 1).detach()

        lmbd_t_list.append(lmbd_tmp)
        a_t_list.append(a_tmp)
        a_hat_t_list.append(a_hat_tmp)
        # gradient descent
        for t in range(self.steps):
            lmbd_updated, a_updated, a_hat_updated = self.update_rule(
                u, m, lmbd_tmp, a_tmp, a_hat_tmp, t)

            a_hat_tmp = a_hat_updated
            a_tmp = a_updated
            lmbd_tmp = lmbd_updated

            lmbd_t_list.append(lmbd_tmp)
            a_t_list.append(a_tmp)
            a_hat_t_list.append(a_hat_tmp)

        # return a_updated
        return a_t_list[1:]

    def update_rule(self, u, m, lmbd, a, a_hat, t):

        grad_a = - u / 2 + (lmbd * soft_sign(torch.sum(a,
            dim=-1) - 1, self.k)).unsqueeze_(-1).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))

        a_hat_updated = a_hat - self.alpha * torch.pow(self.lr_decay_alpha,
            t) * grad
        # the rho needs to be further dealt

        if self.rho_mode=='nn':
            input_features = torch.cat([torch.unsqueeze(a_hat,-1),
                torch.unsqueeze(grad,-1), torch.unsqueeze(u,-1)], -1).view(-1, 3)
            rho = self.rho_net(input_features).view(a_hat.shape)
            a_hat_updated = F.relu(torch.abs(a_hat_updated) - rho)
        elif self.rho_mode=='matrix':
            a_hat_updated = F.relu(
                torch.abs(a_hat_updated) - self.rho_m*self.alpha * torch.pow(self.lr_decay_alpha,t))
        elif self.rho_mode=='nn_pos':
            rho = self.rho_pos_net(self.rho_pos_fea).view(
                a_hat_updated.shape[-2], a_hat_updated.shape[-1])
            a_hat_updated = F.relu(torch.abs(a_hat_updated) - rho)
        else:
            a_hat_updated = F.relu(
                torch.abs(a_hat_updated) - self.rho*self.alpha * torch.pow(self.lr_decay_alpha,t))           

        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_updated = lmbd + self.beta * torch.pow(self.lr_decay_beta, 
            t) * lmbd_grad
        
        return lmbd_updated, a_updated, a_hat_updated
        
class Lag_PP_final(Lag_PP_zero):
    """
    For the update of a and lambda, we use gradient descent with 
    learnable parameters. For the rho, we use neural network to learn
    a position related threshold
    """
    def __init__(self, steps, k, rho_mode='fix'):
        super(Lag_PP_final, self).__init__(steps, k)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = steps
        self.k = k
        self.s = nn.Parameter(torch.Tensor([math.log(9.0)]))
        self.w = nn.Parameter(torch.randn(1))
        self.rho = nn.Parameter(torch.Tensor([1.0]))
        # build the rho network
        # reuse it under every time step
        self.alpha = nn.Parameter(torch.Tensor([0.01]))
        self.beta = nn.Parameter(torch.Tensor([0.1]))
        self.lr_decay_alpha = nn.Parameter(torch.Tensor([0.99]))
        self.lr_decay_beta = nn.Parameter(torch.Tensor([0.99]))
        self.rho_mode = rho_mode


    def forward(self, u, x):
        a_t_list = list()
        a_hat_t_list = list()
        lmbd_t_list = list()

        m = self.constraint_matrix_batch(x) # N*L*L

        u = soft_sign(u - self.s, self.k) * u

        # initialization
        a_hat_tmp = (torch.sigmoid(u)) * soft_sign(u - self.s, self.k).detach()
        a_tmp = self.contact_a(a_hat_tmp, m)
        lmbd_tmp = self.w * F.relu(torch.sum(a_tmp, dim=-1) - 1).detach()

        lmbd_t_list.append(lmbd_tmp)
        a_t_list.append(a_tmp)
        a_hat_t_list.append(a_hat_tmp)
        # gradient descent
        for t in range(self.steps):
            lmbd_updated, a_updated, a_hat_updated = self.update_rule(
                u, m, lmbd_tmp, a_tmp, a_hat_tmp, t)

            a_hat_tmp = a_hat_updated
            a_tmp = a_updated
            lmbd_tmp = lmbd_updated

            lmbd_t_list.append(lmbd_tmp)
            a_t_list.append(a_tmp)
            a_hat_t_list.append(a_hat_tmp)

        # return a_updated
        return a_t_list[1:]

    def update_rule(self, u, m, lmbd, a, a_hat, t):
        grad_a = - u / 2 + (lmbd * soft_sign(torch.sum(a,
            dim=-1) - 1, self.k)).unsqueeze_(-1).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))

        a_hat_updated = a_hat - self.alpha * torch.pow(self.lr_decay_alpha,
            t) * grad
        # the rho needs to be further dealt
        a_hat_updated = F.relu(
            torch.abs(a_hat_updated) - self.rho*self.alpha * torch.pow(self.lr_decay_alpha,t))           

        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_updated = lmbd + self.beta * torch.pow(self.lr_decay_beta, 
            t) * lmbd_grad
        
        return lmbd_updated, a_updated, a_hat_updated

class RNA_SS_e2e(nn.Module):
    def __init__(self, model_att, model_pp):
        super(RNA_SS_e2e, self).__init__()
        self.model_att = model_att
        self.model_pp = model_pp
        
    def forward(self, prior, seq, state):
        u = self.model_att(prior, seq, state)
        map_list = self.model_pp(u, seq)
        return u, map_list



# only using convolutional layers is problematic
# Indeed, if we only use CNN, the spatial information is missing
class ContactNetwork(nn.Module):
    """
    The definition of contact network
    Instantiation: 
        d: the dimension of the hidden dimension of each base
        L: sequence length
    Input: The sequence encoding, the prior knowledge
    Output: The contact map of the input RNA sequence
    """
    def __init__(self, d, L):
        super(ContactNetwork, self).__init__()
        self.d = d
        self.L = L
        # 1d convolution, L*3 to L*d
        self.conv1d1= nn.Conv1d(in_channels=4, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn1 = nn.BatchNorm1d(d)
        self.conv1d2= nn.Conv1d(in_channels=d, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn2 = nn.BatchNorm1d(d)

        # 2d convolution for the matrix representation
        # if possible, we may think of make dilation related the the sequence length
        # and we can consider short-cut link
        self.conv2d1 = nn.Conv2d(in_channels=2*d, out_channels=4*d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn3 = nn.BatchNorm2d(4*d)
        self.conv2d2 = nn.Conv2d(in_channels=4*d, out_channels=2*d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn4 = nn.BatchNorm2d(2*d)

        # 2d convolution for the state
        self.conv2d3 = nn.Conv2d(in_channels=1, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn5 = nn.BatchNorm2d(d)
        self.conv2d4 = nn.Conv2d(in_channels=d, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn6 = nn.BatchNorm2d(d)

        # final convolutional and global pooling, as well as the fc net
        # we may think about multiple paths
        self.conv1 = nn.Conv2d(in_channels=2*d+3, out_channels=3*d, 
            kernel_size=20, padding=19, dilation=2)
        self.bn7 = nn.BatchNorm2d(3*d)
        self.conv2 = nn.Conv2d(in_channels=3*d, out_channels=3*d, 
            kernel_size=20, padding=19, dilation=2)
        self.bn8 = nn.BatchNorm2d(3*d)
        self.conv3 = nn.Conv2d(in_channels=3*d, out_channels=1, 
            kernel_size=20, padding=19, dilation=2)

        self.fc1 = nn.Linear(L*L, L*L)
       

    def forward(self, prior, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))
        seq = F.relu(self.bn2(self.conv1d2(seq))) # d*L
        seq_mat = self.matrix_rep(seq) # 2d*L*L
        seq_mat = F.relu(self.bn3(self.conv2d1(seq_mat)))
        seq_mat = F.relu(self.bn4(self.conv2d2(seq_mat))) # 2d*L*L

        state = nn.functional.one_hot(state.to(torch.int64)-state.min(), 3) # L*L*3
        state = state.permute(0, 3, 1, 2).to(torch.float32) # 3*L*L

        # prior = prior.permute(0, 3, 1, 2).to(torch.float32) # 1*L*L
        # prior = F.relu(self.bn5(self.conv2d3(prior)))
        # prior = F.relu(self.bn6(self.conv2d4(prior))) # d*L*L

        infor = torch.cat([seq_mat, state], 1) # (3d+3)*L*L
        infor = F.relu(self.bn7(self.conv1(infor)))
        # infor = F.relu(self.bn8(self.conv2(infor))) # 3d*L*L
        infor = F.relu(self.conv3(infor)) #1*L*L

        # final dense net
        contact = self.fc1(infor.view(-1, self.L*self.L))
        # contact = infor

        return contact.view(-1, self.L, self.L)
        # return torch.squeeze(infor, 1)


    def matrix_rep(self, x):
        '''
        for each position i,j of the matrix, we concatenate the embedding of i and j
        '''
        x = x.permute(0, 2, 1) # L*d
        L = x.shape[1]
        x2 = x
        x = x.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        x = x.repeat(1, L,1,1)
        x2 = x2.repeat(1, 1, L,1)
        mat = torch.cat([x,x2],-1) # L*L*2d

        # make it symmetric
        # mat_tril = torch.cat(
        #     [torch.tril(mat[:,:, i]) for i in range(mat.shape[-1])], -1)
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2)) # 2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag

        return mat

class ContactNetwork_test(ContactNetwork):
    def __init__(self, d, L):
        super(ContactNetwork_test, self).__init__(d,L)
        self.resnet1d = self._make_layer(ResNetblock, nn.Conv1d, 4, d)
        self.resnet1d_2 = self._make_layer(ResNetblock, nn.Conv1d, 4, d)
        # self.fc1 = nn.Linear(self.d*self.L, self.L*self.L)

        self.conv1d3= nn.Conv1d(in_channels=d+L, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn3 = nn.BatchNorm1d(d)
        self.conv1d4= nn.Conv1d(in_channels=d, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn4 = nn.BatchNorm1d(d)


        self.conv_test = nn.Conv2d(in_channels=3*d, out_channels=1, 
            kernel_size=9, padding=8, dilation=2)
        self.bn_test = nn.BatchNorm2d(1)

        self.position_embedding = nn.Parameter(
            torch.randn(1, d, L, L)
        )

        self.lc = LocallyConnected2d(2*d, 1, L, 1)

    def _make_layer(self, block, conv, layers, plane):
        l = []
        for i in range(layers):
            l.append(block(conv, plane, plane))
        return nn.Sequential(*l)

    def forward(self, prior, seq, state):
        """
        state: L*L*1
        seq: L*4
        """
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))
        infor = F.relu(self.bn2(self.conv1d2(seq))) # d*L
        infor = self.resnet1d(infor) # d*L

        infor = self.matrix_rep(infor) # 2d*L*L

        # position_embeds = self.position_embedding.repeat(infor.shape[0],1,1,1)

        # infor = torch.cat([infor, position_embeds], 1)

        # prior = torch.squeeze(prior, -1)
        # infor = torch.cat([prior, infor], 1) # (d+L)*L
        # infor = F.relu(self.bn3(self.conv1d3(infor)))
        # infor = self.resnet1d_2(infor) # d*L
        # contact = self.fc1(infor.view(-1, self.d*self.L))
        # contact = self.bn_test(self.conv_test(infor))
        contact = self.lc(infor)
        contact = contact.view(-1, self.L, self.L)
        contact = (contact+torch.transpose(contact, -1, -2))/2


        return contact.view(-1, self.L, self.L)




class ContactNetwork_fc(ContactNetwork_test):
    """docstring for ContactNetwork_fc"""
    def __init__(self, d, L):
        super(ContactNetwork_fc, self).__init__(d, L)
        self.fc1 = nn.Linear(self.d*self.L, self.L*self.L)

    def forward(self, prior, seq, state):
        """
        state: L*L*1
        seq: L*4
        """
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))
        infor = F.relu(self.bn2(self.conv1d2(seq))) # d*L
        infor = self.resnet1d(infor) # d*L

        # infor = self.matrix_rep(infor) # 2d*L*L

        # position_embeds = self.position_embedding.repeat(infor.shape[0],1,1,1)

        # infor = torch.cat([infor, position_embeds], 1)

        # prior = torch.squeeze(prior, -1)
        # infor = torch.cat([prior, infor], 1) # (d+L)*L
        # infor = F.relu(self.bn3(self.conv1d3(infor)))
        # infor = self.resnet1d_2(infor) # d*L
        contact = self.fc1(infor.view(-1, self.d*self.L))
        contact = contact.view(-1, self.L, self.L)
        
        # contact = (contact+torch.transpose(contact, -1, -2))/2


        return contact.view(-1, self.L, self.L)

# need to further add the prior knowledge block and the state embedding
class ContactNetwork_ResNet(ContactNetwork):
    def __init__(self, d, L):
        super(ContactNetwork_ResNet, self).__init__(d,L)
        self.resnet1d = self._make_layer(ResNetblock, nn.Conv1d, 4, d)
        self.resnet2d = self._make_layer(ResNetblock, nn.Conv2d, 4, 3*d)
        self.fc1 = nn.Linear(L*L, L*L)
        self.dropout = nn.Dropout(p=0.2)
        self.lc = LocallyConnected2d(3*d, 1, L, 5)

    def _make_layer(self, block, conv, layers, plane):
        l = []
        for i in range(layers):
            l.append(block(conv, plane, plane))
        return nn.Sequential(*l)


    def forward(self, prior, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))
        seq = self.resnet1d(seq) # d*L
        seq_mat = self.matrix_rep(seq) # 2d*L*L

        # deal with state, first embed state
        state = nn.functional.one_hot(state.to(torch.int64)-state.min(), 3) # L*L*3
        state = state.permute(0, 3, 1, 2).to(torch.float32) # 3*L*L

        # prior = prior.permute(0, 3, 1, 2).to(torch.float32) # 1*L*L
        # prior = F.relu(self.bn5(self.conv2d3(prior)))
        # prior = F.relu(self.bn6(self.conv2d4(prior))) # d*L*L

        infor = torch.cat([seq_mat, state], 1) # (2d+3)*L*L
        infor = F.relu(self.bn7(self.conv1(infor)))
        # infor = F.relu(self.bn8(self.conv2(infor))) # 3d*L*L
        infor = self.resnet2d(infor) # 3d*L*L

        # final dense net
        infor = F.relu(self.conv3(infor)) #1*L*L
        # contact = self.fc1(self.dropout(infor.view(-1, self.L*self.L)))
        contact = infor
        # the final locally connected net
        # contact = self.lc(infor)

        return contact.view(-1, self.L, self.L)

# for testing
def testing():
    seq = torch.rand([32, 135, 4])
    contact = torch.zeros([32, 135,135,1], dtype=torch.int32)
    contact[:, :,0]=1
    contact[:, :,1]=-1
    state = torch.zeros([32, 135, 135], dtype=torch.int32)
    m = ContactNetwork_ResNet(d=3, L=135)
    contacts = m(contact, seq, state)
    return contacts
