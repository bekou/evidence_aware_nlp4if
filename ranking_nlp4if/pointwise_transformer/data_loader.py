import os
import torch
import numpy as np
import json
from torch.autograd import Variable


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def tok2int_sent(sentence, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    sent_a, sent_b = sentence
    tokens_a = tokenizer.tokenize(sent_a)

    tokens_b = None
    if sent_b:
        tokens_b = tokenizer.tokenize(sent_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens =  ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    if tokens_b:
        tokens = tokens + tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def tok2int_list(claim_list, tokenizer, max_seq_length, slate_length,labels=[], max_seq_size=-1):

    
    indices_padding=list()
    
    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    
    labels_padding=labels.copy()
    mask_evidences_padding=list()
    
    for claim_i,src_list in  enumerate(claim_list):
        inp_padding_tokens = list()
        msk_padding_tokens = list()
        seg_padding_tokens = list()
        ind_padding_evidences=list()
        mask_evidences=list()
        for evidence_i, sent in enumerate(src_list):
            #print ()
            #print (claim_i)
            #print (evidence_i)
            
            
            input_ids, input_mask, input_seg = tok2int_sent(sent, tokenizer, max_seq_length)
            
            
            inp_padding_tokens.append(input_ids)
            msk_padding_tokens.append(input_mask)
            seg_padding_tokens.append(input_seg)
            
            ind_padding_evidences.append(evidence_i)
            mask_evidences.append(1)
        
        padding_evidences = [-1] * (slate_length - len(inp_padding_tokens)) # pad the labels and indices tensors
        padding_evidences_mask = [0] * (slate_length - len(inp_padding_tokens)) # pad the mask evidence tensor
        padding = [0] * (max_seq_length)
        if len(inp_padding_tokens) < slate_length: #padding word level
               if labels!=[]:
                    labels_padding[claim_i]+=padding_evidences
               ind_padding_evidences+=padding_evidences
               mask_evidences+=padding_evidences_mask
               for i in range(slate_length - len(inp_padding_tokens)):
                        #word_ids.append(word_dictionary['<pad>'])#<pad>
                        inp_padding_tokens.append(padding)
                        msk_padding_tokens.append(padding)
                        seg_padding_tokens.append(padding)
        elif len(inp_padding_tokens) > slate_length: #cut redundant
                ind_padding_evidences=ind_padding_evidences[:slate_length] # not necessary
                mask_evidences=mask_evidences[:slate_length] # not necessary
                if labels!=[]:
                    labels_padding[claim_i]=labels_padding[claim_i][:slate_length]
                inp_padding_tokens=inp_padding_tokens[:slate_length]
                msk_padding_tokens=msk_padding_tokens[:slate_length]
                seg_padding_tokens=seg_padding_tokens[:slate_length]
            
        #print (labels[claim_i])
        
        inp_padding.append(inp_padding_tokens)
        msk_padding.append(msk_padding_tokens)
        seg_padding.append(seg_padding_tokens)
        
        indices_padding.append(ind_padding_evidences)
        mask_evidences_padding.append(mask_evidences)
    '''
    padding = [0] * (max_seq_length)
    if len(inp_padding) < slate_length: #padding word level
           for i in range(slate_length - len(inp_padding)):
                    #word_ids.append(word_dictionary['<pad>'])#<pad>
                    inp_padding.append(padding)
                    msk_padding.append(padding)
                    seg_padding.append(padding)
    elif len(inp_padding) > slate_length: #cut redundant
            inp_padding=inp_padding[:slate_length]
            msk_padding=msk_padding[:slate_length]
            seg_padding=seg_padding[:slate_length]
    '''            
    
    
        #if max_seq_size != -1:
        #    inp_padding = inp_padding[:max_seq_size]
        #    msk_padding = msk_padding[:max_seq_size]
        #    seg_padding = seg_padding[:max_seq_size]
        #    inp_padding += ([[0] * max_seq_length] * (max_seq_size - len(inp_padding)))
        #    msk_padding += ([[0] * max_seq_length] * (max_seq_size - len(msk_padding)))
        #    seg_padding += ([[0] * max_seq_length] * (max_seq_size - len(seg_padding)))
    return inp_padding, msk_padding, seg_padding,labels_padding,indices_padding,mask_evidences_padding


class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, args, test=False, cuda=True, batch_size=64):
        self.cuda = cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.slate_length=args.slate_length
        self.threshold = args.threshold
        self.data_path = data_path
        self.test = test
        examples = self.read_file(data_path)
        self.examples = examples
        self.total_num = len(examples)
        if self.test:
            self.total_num = len(self.examples)
            self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        else:
            self.total_step = self.total_num / batch_size
            #self.shuffle()
        self.step = 0



    def read_file(self, data_path):
        examples = list()
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                data = json.loads(line)
                #claim_id= data["id"]
                claim = data["claim"]
                evidences = data["evidence"]
                pos_evi = list()
                neg_evi = list()
                for evidence in evidences:
                    if (evidence[3] == 1 or evidence[3] == 2) and evidence[2].strip() != "":
                        pos_evi.append(evidence)
                    elif evidence[3] == 0  and evidence[2].strip() != "":
                        neg_evi.append(evidence)
                total_triples = pos_evi
                pos_num = len(pos_evi)
                neg_num = self.evi_num * pos_num
                np.random.shuffle(neg_evi)
                neg_evi = neg_evi[:neg_num]
                total_triples += neg_evi
                #print ("h1")
                #print (claim)
                #print (total_triples)
                #print ("h2")
                claim_evid_list = list()
                for triple in total_triples:                
                    
                    if triple[3] == 1 or triple[3] == 2:
                        #examples.append([claim, triple[2], 1])
                        claim_evid_list.append([claim, triple[2], 1])
                        #print ([claim, triple[2], 1])
                    elif triple[3] == 0:
                        claim_evid_list.append([claim, triple[2], 0])
                        #examples.append([claim, triple[2], 0])
                        #print ([claim, triple[2], 0])
                if len(claim_evid_list)>0:        
                        examples.append(claim_evid_list)
                        
                
                #print ()
        return examples


    def shuffle(self):
        np.random.shuffle(self.examples)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''
        if self.step < self.total_step:
            examples = self.examples[self.step * self.batch_size : (self.step+1)*self.batch_size]
            
            #print ("xxxxx")
            
            #print (examples)
            
            #print (len(examples))
            
            
            inputs = list()
            labels = list()
            for example in examples:
            
                #print ("example")
                #print (example)
                inputs_internal=list()
                labels_internal=list()
                for sentence in example:
                    inputs_internal.append([sentence[0], sentence[1]])
                    labels_internal.append(sentence[2])
                #print([example[0], example[1],example[2]])
                inputs.append(inputs_internal)
                labels.append(labels_internal)            
            #print ('inputs')    
            #print (inputs)
            #print ('labels')
            #print (labels)
            
                   

            
            inp, msk_tk, seg,lab,ind,msk_ev = tok2int_list(inputs, self.tokenizer, self.max_len,self.slate_length,labels)

            '''
            print (len(inp))
            print (len(inp[0]))
            print (len(inp[0][0]))
            print (len(inp[1]))
            
            
            print (inp[0])
            
            print ("x2x2x2x2x2")
            
            print (lab)
            
            print (ind)
            print (len(lab))
            '''

            inp_tensor = Variable(
                torch.LongTensor(inp))
                
            #print (inp_tensor)
            #print (inp_tensor.shape)

            msk_tk_tensor = Variable(
                torch.LongTensor(msk_tk))
            seg_tensor = Variable(
                torch.LongTensor(seg))
            lab_tensor = Variable(
                torch.FloatTensor(lab))
            ind_tensor = Variable(
                torch.LongTensor(ind))
            msk_ev_tensor = Variable(
                torch.FloatTensor(msk_ev))
            
            '''            
            print (lab_tensor)
            print (lab_tensor.shape)
            print (ind_tensor)
            print (ind_tensor.shape)
            
            
            torch.set_printoptions(edgeitems=40)    
            print (inp_tensor)
            print (inp_tensor.shape)
            
            print (msk_tensor)
            print (msk_tensor.shape)
            
            print (seg_tensor)
            print (seg_tensor.shape)
            '''
            
            if self.cuda:
                inp_tensor = inp_tensor.cuda()
                msk_tk_tensor = msk_tk_tensor.cuda()
                seg_tensor = seg_tensor.cuda()
                lab_tensor = lab_tensor.cuda()
                ind_tensor = ind_tensor.cuda()
                msk_ev_tensor = msk_ev_tensor.cuda()
            self.step += 1
            return inp_tensor, msk_tk_tensor, seg_tensor, lab_tensor,ind_tensor,msk_ev_tensor
        else:
            self.step = 0
            if not self.test:
                examples = self.read_file(self.data_path)
                self.examples = examples
                self.shuffle()
            raise StopIteration()

class DataLoaderTest(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, args, cuda=True, batch_size=64):
        self.cuda = cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        
        self.threshold = args.threshold
        self.data_path = data_path
        inputs, ids, evi_list,self.slate_length = self.read_file(data_path)
        self.inputs = inputs
        self.ids = ids
        self.evi_list = evi_list

        self.total_num = len(inputs)
        self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        self.step = 0


    def read_file(self, data_path):
        inputs = list()
        ids = list()
        evi_list = list()
        #c=0
        evi_numbers=list()
        #print (evi_numbers)
        #print (evi_numbers)
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                instance = json.loads(line.strip())
                claim = instance['claim']
                id = instance['id']
                
                claim_ids=list()
                inputs_claim=list()
                evidences_claim=list()
                #if len(instance['evidence'])>150:
                #    c+=1
                #    print (len(instance['evidence']))
                evi_numbers.append(len(instance['evidence']))
                for evidence in instance['evidence']:
                
                    #print ('---')
                    #print ('id')
                    #print (id)
                    claim_ids.append(id)
                    #print ('[claim, evidence[2]]')
                    #print ([claim, evidence[2]])
                    inputs_claim.append([claim, evidence[2]])
                    #print ('evidence')
                    #print (evidence)
                    evidences_claim.append(evidence)
                inputs.append(inputs_claim)
                ids.append(claim_ids)
                evi_list.append(evidences_claim)
        #print (evi_numbers)
        #print (max(evi_numbers))
        #import numpy as np
        #print (c)

        #p = np.percentile(evi_numbers, 95)
        #print (p)
        return inputs, ids, evi_list,max(evi_numbers)


    def shuffle(self):
        np.random.shuffle(self.examples)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''
        if self.step < self.total_step:
            inputs = self.inputs[self.step * self.batch_size : (self.step+1)*self.batch_size]
            ids = self.ids[self.step * self.batch_size: (self.step + 1) * self.batch_size]
            evi_list = self.evi_list[self.step * self.batch_size: (self.step + 1) * self.batch_size]
            '''
            print ("inputs")
            print (inputs)
            print ("ids")
            print (ids)
            print ("evi_list")
            print (evi_list)
            '''
            #inp_padding, msk_padding, seg_padding,labels_padding,indices_padding,mask_evidences_padding
            #inp, msk, seg = tok2int_list(inputs, self.tokenizer, self.max_len, -1)
            
            inp, msk_tk, seg,_,ind,msk_ev,slate_batch = tok2int_list_test(inputs, self.tokenizer, self.max_len)
            
            
            inp_tensor_input = Variable(
                torch.LongTensor(inp))
            msk_tk_tensor_input = Variable(
                torch.LongTensor(msk_tk))
            seg_tensor_input = Variable(
                torch.LongTensor(seg))
            ind_tensor = Variable(
                torch.LongTensor(ind))
            msk_ev_tensor = Variable(
                torch.FloatTensor(msk_ev))    
                
            
            
            if self.cuda:
                inp_tensor_input = inp_tensor_input.cuda()
                msk_tk_tensor_input = msk_tk_tensor_input.cuda()
                seg_tensor_input = seg_tensor_input.cuda()
                ind_tensor = ind_tensor.cuda()
                msk_ev_tensor = msk_ev_tensor.cuda()
            self.step += 1
            return inp_tensor_input, msk_tk_tensor_input, seg_tensor_input,ind_tensor,msk_ev_tensor, ids, evi_list,slate_batch
        else:
            self.step = 0
            raise StopIteration()
#define the function#
def find_max_list(list):
    list_len = [len(i) for i in list]
    #print(max(list_len))
    return max(list_len)



def tok2int_list_test(claim_list, tokenizer, max_seq_length,labels=[]):

    
    indices_padding=list()
    
    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    
    labels_padding=labels.copy()
    mask_evidences_padding=list()
    
    #print output#
    #print (len(claim_list))
    #print (find_max_list(claim_list))
    #print (find_longest_list(claim_list))
    
    slate_in_batch=find_max_list(claim_list)
    
    slate_length=slate_in_batch
    
    
    for claim_i,src_list in  enumerate(claim_list):
    
        
        inp_padding_tokens = list()
        msk_padding_tokens = list()
        seg_padding_tokens = list()
        ind_padding_evidences=list()
        mask_evidences=list()
        for evidence_i, sent in enumerate(src_list):
            #print ()
            #print (claim_i)
            #print (evidence_i)
            
            
            input_ids, input_mask, input_seg = tok2int_sent(sent, tokenizer, max_seq_length)
            
            
            inp_padding_tokens.append(input_ids)
            msk_padding_tokens.append(input_mask)
            seg_padding_tokens.append(input_seg)
            
            ind_padding_evidences.append(evidence_i)
            mask_evidences.append(1)
        
        padding_evidences = [-1] * (slate_length - len(inp_padding_tokens)) # pad the labels and indices tensors
        padding_evidences_mask = [0] * (slate_length - len(inp_padding_tokens)) # pad the mask evidence tensor
        padding = [0] * (max_seq_length)
        if len(inp_padding_tokens) < slate_length: #padding word level
               if labels!=[]:
                    labels_padding[claim_i]+=padding_evidences
               ind_padding_evidences+=padding_evidences
               mask_evidences+=padding_evidences_mask
               for i in range(slate_length - len(inp_padding_tokens)):
                        #word_ids.append(word_dictionary['<pad>'])#<pad>
                        inp_padding_tokens.append(padding)
                        msk_padding_tokens.append(padding)
                        seg_padding_tokens.append(padding)
        elif len(inp_padding_tokens) > slate_length: #cut redundant
                ind_padding_evidences=ind_padding_evidences[:slate_length] # not necessary
                mask_evidences=mask_evidences[:slate_length] # not necessary
                if labels!=[]:
                    labels_padding[claim_i]=labels_padding[claim_i][:slate_length]
                inp_padding_tokens=inp_padding_tokens[:slate_length]
                msk_padding_tokens=msk_padding_tokens[:slate_length]
                seg_padding_tokens=seg_padding_tokens[:slate_length]
            
        #print (labels[claim_i])
        
        inp_padding.append(inp_padding_tokens)
        msk_padding.append(msk_padding_tokens)
        seg_padding.append(seg_padding_tokens)
        
        indices_padding.append(ind_padding_evidences)
        mask_evidences_padding.append(mask_evidences)
    '''
    padding = [0] * (max_seq_length)
    if len(inp_padding) < slate_length: #padding word level
           for i in range(slate_length - len(inp_padding)):
                    #word_ids.append(word_dictionary['<pad>'])#<pad>
                    inp_padding.append(padding)
                    msk_padding.append(padding)
                    seg_padding.append(padding)
    elif len(inp_padding) > slate_length: #cut redundant
            inp_padding=inp_padding[:slate_length]
            msk_padding=msk_padding[:slate_length]
            seg_padding=seg_padding[:slate_length]
    '''            
    
    
        #if max_seq_size != -1:
        #    inp_padding = inp_padding[:max_seq_size]
        #    msk_padding = msk_padding[:max_seq_size]
        #    seg_padding = seg_padding[:max_seq_size]
        #    inp_padding += ([[0] * max_seq_length] * (max_seq_size - len(inp_padding)))
        #    msk_padding += ([[0] * max_seq_length] * (max_seq_size - len(msk_padding)))
        #    seg_padding += ([[0] * max_seq_length] * (max_seq_size - len(seg_padding)))
    return inp_padding, msk_padding, seg_padding,labels_padding,indices_padding,mask_evidences_padding,slate_in_batch