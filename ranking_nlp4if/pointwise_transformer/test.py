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


def save_to_file(all_predict, outpath):
    with open(outpath, "w") as out:
        for key, values in all_predict.items():
            sorted_values = sorted(values, key=lambda x:x[-1], reverse=True)
            data = json.dumps({"id": key, "evidence": sorted_values[:5]})
            out.write(data + "\n")



def eval_model(models, validset_reader):
    model,model_transf=models
    model.eval()
    model_transf.eval()
    all_predict = dict()
    
    for inp_tensor_input, msk_tk_tensor_input, seg_tensor_input,ind_tensor,msk_ev_tensor, ids, evi_list,slate_batch in validset_reader:
       
        bert_output = model(inp_tensor_input, msk_tk_tensor_input, seg_tensor_input,ind_tensor,msk_ev_tensor,args.batch_size,slate_batch)
            
        
        mask = torch.as_tensor(msk_ev_tensor == 0) 
        
            
        scores=model_transf(bert_output, mask, ind_tensor)
        
        scores = scores.tolist()
        assert len(scores) == len(evi_list)
        for b in range(len(ids)):
            for i in range(len(ids[b])):
                
            
                if ids[b][i] not in all_predict:
                    all_predict[ids[b][i]] = []
                
                all_predict[ids[b][i]].append(evi_list[b][i] + [scores[b][i]])
        
        
    return all_predict
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', help='train path')
    parser.add_argument('--name', help='train path')
    parser.add_argument("--batch_size", default=1, type=int, help="Total batch size for training.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument('--gpu', default=0, help='Cuda GPU id.')
    parser.add_argument('--bert_pretrain', required=True)
    #parser.add_argument("--slate_length", default=400, type=int, help="Total batch size for training.")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
    parser.add_argument("--max_len", default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
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
    validset_reader = DataLoaderTest(args.test_path, tokenizer, args, batch_size=args.batch_size)  

    logger.info('initializing estimator model')
    bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)
    
    
    args.slate_length=validset_reader.slate_length
    model = inference_model(bert_model, args)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    
    model = model.cuda()
    
    
    config = Config.from_json(file_path+'/scripts/local_config.json')
    model_transf = make_model(n_features=args.bert_hidden_dim, **asdict(config.model, recurse=False))   
    
    model_transf.load_state_dict(torch.load(args.checkpoint)['model_transf'])
    
    model_transf = model_transf.cuda()
    
    
    logger.info('Start eval!')
    save_path = args.outdir + "/" + args.name
    
    models=[model,model_transf]
    with torch.no_grad():

        predict_dict = eval_model(models, validset_reader)
    save_to_file(predict_dict, save_path)