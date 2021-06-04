import random, os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from models import inference_model
from data_loader import DataLoader, DataLoaderTest
from bert_model import BertForSequenceEncoder
from torch.nn import NLLLoss
import logging
import json
import torch.nn as nn
from sklearn.metrics import accuracy_score,f1_score,classification_report

import os, sys

file_path=os.path.abspath("../../allRank-kgat/")
sys.path.append(os.path.dirname(file_path+"//"))
import allrank.models.losses as losses
from allrank.models.model import make_model
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.config import Config
from attr import asdict
from functools import partial
import allrank.models.metrics as metrics_module


logger = logging.getLogger(__name__)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    if 1.0 - x >= 0.0:
        return 1.0 - x
    return 0.0

def correct_prediction(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

def eval_model(model, validset_reader):
    with torch.no_grad():

        model,model_transf=models
        model.eval()
        model_transf.eval()
        
        
        eval_scores=[]
        
        
        for inp_tensor, msk_tensor, seg_tensor, label_tensor,ind_tensor,msk_ev_tensor in validset_reader:
            bert_output = model(inp_tensor, msk_tensor, seg_tensor,ind_tensor,msk_ev_tensor,args.train_batch_size)
                
            mask = (label_tensor == PADDED_Y_VALUE)
            
                
            output=model_transf(bert_output, mask, ind_tensor)
            ndcg=metrics_module.ndcg(output,label_tensor,ats=[5])
            
            eval_scores.append(ndcg)
            
        mean_ndcg=torch.mean(torch.cat(eval_scores)).cpu().numpy()
        
        
        return mean_ndcg#dev_accuracy,f1


def train_model(models, args, trainset_reader, validset_reader):
    model,model_transf=models
    save_path = args.outdir + '/model'
    
    best_ndcg=0.0
    running_loss = 0.0
    t_total = int(
        trainset_reader.total_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    param_optimizer = list(model.named_parameters())+list(model_transf.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                        warmup=args.warmup_proportion,
                         t_total=t_total)
    global_step = 0
    
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    
    for epoch in range(int(args.num_train_epochs)):
        optimizer.zero_grad()
        for inp_tensor, msk_tensor, seg_tensor, label_tensor,ind_tensor,msk_ev_tensor in trainset_reader:
            model.train()
            model_transf.train()
            
            bert_output = model(inp_tensor, msk_tensor, seg_tensor,ind_tensor,msk_ev_tensor,args.train_batch_size)
            
            mask = (label_tensor == PADDED_Y_VALUE)
            
            output=model_transf(bert_output, mask, ind_tensor)
            
            loss=loss_func(output,label_tensor)
            
            running_loss += loss.item()
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                logger.info('Epoch: {0}, Step: {1}, Loss: {2}'.format(epoch, global_step, (running_loss / global_step)))
            if global_step % (args.eval_step * args.gradient_accumulation_steps) == 0:
                logger.info('Start eval!')
                eval_ndcg = eval_model(models, validset_reader)
                logger.info('Dev ndcg@5: {0}'.format(eval_ndcg))
                if eval_ndcg >= best_ndcg:
                    best_ndcg = eval_ndcg
                    torch.save({'epoch': epoch,
                                'model': model.state_dict(),'model_transf': model_transf.state_dict()}, save_path + ".best.pt")
                    logger.info("Saved best epoch {0}, best ndcg {1}".format(epoch, best_ndcg))
               
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument('--gpu', default=0, help='Cuda GPU id.')
    parser.add_argument("--train_batch_size", default=2, type=int, help="Total batch size for training.")
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=2, type=int, help="Total batch size for predictions.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--slate_length", type=int, default=10)
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
    parser.add_argument("--max_len", default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--eval_step", default=2000, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--bert_pretrain', required=True)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] =str(args.gpu)  

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logger.info('Start training!')

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
    logger.info("loading training set")
    trainset_reader = DataLoader(args.train_path, tokenizer, args, batch_size=args.train_batch_size)
    logger.info("loading validation set")
    validset_reader = DataLoader(args.valid_path, tokenizer, args, batch_size=args.valid_batch_size, test=True)

    logger.info('initializing estimator model')
    bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)
    bert_model = bert_model.cuda()
    model = inference_model(bert_model, args)
    model = model.cuda()
    
    config = Config.from_json(file_path+'/scripts/local_config.json')
    model_transf = make_model(n_features=args.bert_hidden_dim, **asdict(config.model, recurse=False))   
    model_transf= model_transf.cuda()
    models=[model,model_transf]
    train_model(models, args, trainset_reader, validset_reader)