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

class classifier(nn.Module):
    def __init__(self, bert_model, args):
        super(classifier, self).__init__()
        
        self.dropout = nn.Dropout(0.1)
        self.features = 128
        self.num_labels = 2
        
        self.proj_match = nn.Linear(self.features, self.num_labels)
        
        

    def forward(self, inputs):
        
        inputs = self.dropout(inputs)
        #print ( self.proj_match(inputs).shape)
        score = self.proj_match(inputs).squeeze(-1)
        #print (score.shape)
        
        prob = F.log_softmax(score, dim=1)
        #print ((prob).shape)
        #print (prob)
        return prob


class inference_model(nn.Module):
    def __init__(self, bert_model, args):
        super(inference_model, self).__init__()
        self.bert_hidden_dim = args.bert_hidden_dim
        self.dropout = nn.Dropout(args.dropout)
        self.max_len = args.max_len
        self.num_labels = args.num_labels
        self.pred_model = bert_model
        #self.proj_hidden = nn.Linear(self.bert_hidden_dim, 128)
        
        self.proj_match = nn.Linear(self.bert_hidden_dim, 128)


    def forward(self, inp_tensor, msk_tensor, seg_tensor):
        _, inputs = self.pred_model(inp_tensor, msk_tensor, seg_tensor)
        inputs = self.dropout(inputs)
        score = self.proj_match(inputs).squeeze(-1)
        score = torch.tanh(score)
        return score