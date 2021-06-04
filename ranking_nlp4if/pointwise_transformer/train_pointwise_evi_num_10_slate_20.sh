python train_pointwise.py --outdir ../checkpoint/pretrain/pretrain_transformer_evi_num_10_slate_20/ \
--train_path ../data/all_train.json \
--valid_path ../data/all_dev.json \
--bert_pretrain ../bert_base \
--evi_num 10 \
--slate_length 20 \
--gpu 0