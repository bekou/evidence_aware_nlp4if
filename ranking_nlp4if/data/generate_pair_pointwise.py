import os
import json
import argparse
import numpy as np
import random
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile')
    parser.add_argument('--outfile')
    parser.add_argument('--slate_length',type=int,default=-1, help="Total number of evidences positive and negative.")
    parser.add_argument('--evi_num',type=int,default=5, help="Number of negatives per positive evidence.")
    parser.add_argument('--shuffle',type=int,default=1)
    
    
    
    args = parser.parse_args()
    
    #print (args.slate_length)
    #print (args.evi_num)
    #print (args.shuffle)
    pairs = list()
    
    #slate_length=10
    #evi_num=5
    #max_pairs_per_claim=10
    with open(args.infile) as f:
            
            for line in f:
                data = json.loads(line)
                
                
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
                neg_num = args.evi_num * pos_num
                np.random.shuffle(neg_evi)
                neg_evi = neg_evi[:neg_num]
                total_triples += neg_evi
                
                if args.slate_length>0:
                    #print ("here")
                    total_triples= total_triples[0:args.slate_length]
                
                for evidence in total_triples:
                    if evidence[3] == 1:
                        for evidence_ in total_triples:
                            if evidence_[3] == 0:
                                sent1 = " ".join(evidence[2].strip().split())
                                sent2 = " ".join(evidence_[2].strip().split())
                                if sent1 != "" and sent2 != "":
                                    pairs.append([claim, evidence[0], sent1, evidence_[0], sent2])
                
                
               
                        
    with open(args.outfile, "w") as out:
        if args.shuffle==1:
            #print ("here2")
            np.random.shuffle(pairs)
        for pair in pairs:
            out.write("\t".join(pair) + "\n")