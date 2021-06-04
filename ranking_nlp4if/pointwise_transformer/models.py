import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU
from bert_model import BertForSequenceEncoder

from torch.nn import BatchNorm1d, Linear, ReLU
from bert_model import BertForSequenceEncoder
from torch.autograd import Variable
import numpy as np




def kernal_mus(n_kernels):
    """
    get the mu for each guassian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

class inference_model(nn.Module):
    def __init__(self, bert_model, args):
        super(inference_model, self).__init__()
        self.bert_hidden_dim = args.bert_hidden_dim
        self.dropout = nn.Dropout(args.dropout)
        self.max_len = args.max_len
        self.slate_length=args.slate_length
        self.num_labels = args.num_labels
        self.pred_model = bert_model
        #self.proj_hidden = nn.Linear(self.bert_hidden_dim, 128)
        #self.proj_match = nn.Linear(self.bert_hidden_dim, self.num_labels)


    def forward(self, inp_tensor, msk_tensor, seg_tensor,ind_tensor,msk_ev_tensor,batch_size,slate_length=-1):
        
        #print (inp_tensor)
        if slate_length==-1:
            slate_length=self.slate_length
            
        #print (slate_length)
        
        inp_tensor=inp_tensor.view(batch_size,slate_length,self.max_len)
        inp_tensor,msk_tensor,seg_tensor,msk_ev_tensor=inp_tensor.view(batch_size*slate_length,self.max_len), \
                                                        msk_tensor.view(batch_size*slate_length,self.max_len), \
                                                        seg_tensor.view(batch_size*slate_length,self.max_len), \
                                                        msk_ev_tensor.view(batch_size*slate_length)
        #print (inp_tensor.view(-1,self.slate_length,self.max_len))
        
        #print (inp_tensor.shape)
        _, inputs = self.pred_model(inp_tensor, msk_tensor, seg_tensor)
        '''
        #print ("_")
        #print (_)
        #print (_.shape)
        
        
        print (inputs)
        print (inputs.shape)
        
        #print (msk_tensor)
        #print (msk_tensor.shape)
        
        #seg_tensor
        print (msk_ev_tensor)
        print (msk_ev_tensor.shape)
        
        #print (inputs*msk_ev_tensor)
        '''
        inputs=torch.einsum('ij, i -> ij', inputs,msk_ev_tensor)
        #print (inputs.shape)
        
        inputs,msk_ev_tensor=inputs.view(batch_size,slate_length,self.bert_hidden_dim), \
                                                        msk_ev_tensor.view(batch_size,slate_length)
                                                        
        
        #print (inputs)
        #print (inputs.shape)
        #print (msk_ev_tensor)
        #print (msk_ev_tensor.shape)
        '''
        #x=torch.matmul(msk_ev_tensor,inputs)
        
        print (inputs)
        print (inputs.shape)
        
        print (x)
        print (x.shape)
        '''
        
        
        inputs = self.dropout(inputs)
        
        
        '''
        score = self.proj_match(inputs).squeeze(-1)
        #print ("score")
        print (self.num_labels)
        print (score)
        print (score.shape)
        prob = F.softmax(score, dim=-1)
        #print ("prob")
        #print (prob)
        '''
        return inputs
