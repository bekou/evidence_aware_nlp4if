python train.py --outdir ../checkpoint/pw_evi_num_10_slate_20_kgat/ \
--train_path ../data/bert_train_pw_evi_num_10_slate_20.json \
--valid_path ../data/bert_dev_pw_evi_num_10_slate_20.json \
--bert_pretrain ../bert_base \
--train_batch_size 4 \
--classifier kgat \
--gpu 0

python test.py --outdir ./output/pw_evi_num_10_slate_20_kgat/ \
--test_path ../data/bert_eval_pw_evi_num_10_slate_20.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/pw_evi_num_10_slate_20_kgat/model.best.pt \
--name dev.json \
--classifier kgat \
--gpu 0

python test.py --outdir ./output/pw_evi_num_10_slate_20_kgat/ \
--test_path ../data/bert_test_pw_evi_num_10_slate_20.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/pw_evi_num_10_slate_20_kgat/model.best.pt \
--name test.json \
--classifier kgat \
--gpu 0

python fever_score_test.py --predicted_labels ./output/pw_evi_num_10_slate_20_kgat/dev.json  --predicted_evidence ../data/bert_eval_pw_evi_num_10_slate_20.json --actual ../data/dev_eval.json
#python $VSC_DATA/projects/KernelGAT/kgat/fever_score_test.py --predicted_labels $VSC_DATA/projects/KernelGAT/output/kgat/npretrain/test.json  --predicted_evidence $VSC_DATA/projects/KernelGAT/data/bert_test.json --actual ../data/test_eval.json
