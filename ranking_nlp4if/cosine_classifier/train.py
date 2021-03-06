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

from models import inference_model,classifier
from data_loader import DataLoader, DataLoaderTest
from bert_model import BertForSequenceEncoder
from torch.nn import NLLLoss
import logging
import json
import torch.nn as nn

logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    if 1.0 - x >= 0.0:
        return 1.0 - x
    return 0.0

'''
def correct_prediction(prob_pos, prob_neg,prob_anc):
    correct = 0.0
    prob_pos = prob_pos#.view(-1)#.tolist()
    prob_neg = prob_neg#.view(-1)#.tolist()
    #print (prob_pos.shape)
    #print (torch.norm(prob_pos,dim=1).shape)
    #norm_pos=torch.norm(prob_pos,dim=1)
    #norm_neg=torch.norm(prob_neg,dim=1)
    assert len(prob_pos) == len(prob_neg)
    assert len(prob_pos) == len(prob_anc)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_pos = cos(prob_anc,prob_pos)
    cos_neg = cos(prob_anc,prob_neg)
    
    #print (cos_pos)
    #print (cos_neg)
    for step in range(len(cos_pos)):
        if cos_pos[step] > cos_neg[step]:
            correct += 1
    return correct
'''               
def correct_prediction(output, labels):
    #print (output)
    #print (output.max(1))
    preds = output.max(1)[1].type_as(labels)
    #print (preds)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

def eval_model(models, validset_reader):
    model=models[0]
    model2=models[1]
    model.eval()
    model2.eval()
    correct_pred = 0.0
    for inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, inp_tensor_neg, msk_tensor_neg, seg_tensor_neg,inp_tensor_anc, msk_tensor_anc, seg_tensor_anc in validset_reader:
                                                                        
        prob_pos = model(inp_tensor_pos, msk_tensor_pos, seg_tensor_pos)
        prob_neg = model(inp_tensor_neg, msk_tensor_neg, seg_tensor_neg)
        
        scores=torch.cat((prob_pos,prob_neg),0).cuda()
        labels_pos = torch.ones(prob_pos.shape[0])
        labels_neg = torch.zeros(prob_neg.shape[0])
        lab_tensor = torch.cat((labels_pos,labels_neg),0).cuda().long()
        
        prob = model2(scores)
        
        #print (prob.shape)
        #print (lab_tensor.shape)
        
        correct_pred += correct_prediction(prob, lab_tensor)
    dev_accuracy = correct_pred / (validset_reader.total_num*2)
    return dev_accuracy   
'''
def eval_model(model, validset_reader):
    model.eval()
    correct_pred = 0.0
    for inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, inp_tensor_neg, msk_tensor_neg, seg_tensor_neg,inp_tensor_anc, msk_tensor_anc, seg_tensor_anc in validset_reader:
        prob_anc = model(inp_tensor_anc, msk_tensor_anc, seg_tensor_anc)
        prob_pos = model(inp_tensor_pos, msk_tensor_pos, seg_tensor_pos)
        prob_neg = model(inp_tensor_neg, msk_tensor_neg, seg_tensor_neg)
        
        
        
        correct_pred += correct_prediction(prob_pos, prob_neg,prob_anc)
    dev_accuracy = correct_pred / validset_reader.total_num
    return dev_accuracy

'''


def train_model(models, args, trainset_reader, validset_reader):
    model=models[0]
    model2=models[1]
    save_path = args.outdir + '/model'
    best_acc = 0.0
    running_loss = 0.0
    running_loss_1 = 0.0
    running_loss_2 = 0.0
    t_total = int(
        trainset_reader.total_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    param_optimizer = list(model.named_parameters())+list(model2.named_parameters())
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
    crit = nn.CosineEmbeddingLoss(margin=0)
    for epoch in range(int(args.num_train_epochs)):
        optimizer.zero_grad()
        for inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, inp_tensor_neg, msk_tensor_neg, seg_tensor_neg,inp_tensor_anc, msk_tensor_anc, seg_tensor_anc in trainset_reader:
            if global_step<1e9:
                model.train()
                model2.train()
                score_pos = model(inp_tensor_pos, msk_tensor_pos, seg_tensor_pos)
                score_neg = model(inp_tensor_neg, msk_tensor_neg, seg_tensor_neg)
                #print (inp_tensor_pos.shape)
                #print (score_pos.shape)
                #print (score_neg.shape)
                
                label = torch.ones(score_pos.shape[0]) * -1
                if args.cuda:
                    label = label.cuda()
                #print (label)
                loss = crit(score_pos, score_neg, Variable(label, requires_grad=False))
                
                running_loss_1 += loss.item()
                
                scores=torch.cat((score_pos,score_neg),0).cuda()
                labels_pos = torch.ones(score_pos.shape[0])
                labels_neg = torch.zeros(score_neg.shape[0])
                lab_tensor = torch.cat((labels_pos,labels_neg),0).cuda().long()
                #print (labels_pos.shape)
                #print (labels_neg.shape)
                #print (lab_tensor)
                    
                prob = model2(scores)
                ###
                loss2 = F.nll_loss(prob, lab_tensor)
                running_loss_2 += loss2.item()
                
                
                (loss+loss2).backward()
                global_step += 1
                if global_step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    logger.info('Epoch: {0}, Step: {1}, Loss: {2}, Loss 1: {3}, Loss 2: {4}'.format(epoch, global_step, (running_loss_1+running_loss_2) / global_step,running_loss_1/global_step,running_loss_2/global_step))
                if global_step % (args.eval_step * args.gradient_accumulation_steps) == 0:
                    logger.info('Start eval!')
                    eval_acc = eval_model(models, validset_reader)
                    logger.info('Dev acc: {0}'.format(eval_acc))
                    if eval_acc >= best_acc:
                        best_acc = eval_acc
                        torch.save({'epoch': epoch,
                                    'model': model.state_dict(),'model2': model2.state_dict()}, save_path + ".best.pt")
                        logger.info("Saved best epoch {0}, best acc {1}".format(epoch, best_acc))
            else:
                 raise ValueError('Time to run next job.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument('--gpu', default=0, help='Cuda GPU id.')
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
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
    
    model2 = classifier(bert_model, args)
    model2 = model2.cuda()
    models=[model,model2]
    
    train_model(models, args, trainset_reader, validset_reader)