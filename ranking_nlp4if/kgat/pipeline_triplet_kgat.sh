python train.py --outdir ../checkpoint/triplet_kgat/ \
--train_path ../data/bert_train_triplet.json \
--valid_path ../data/bert_dev_triplet.json \
--bert_pretrain ../bert_base \
--classifier kgat \
--gpu 0

python test.py --outdir ./output/triplet_kgat/ \
--test_path ../data/bert_eval_triplet.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/triplet_kgat/model.best.pt \
--name dev.json \
--classifier kgat \
--gpu 0

python test.py --outdir ./output/triplet_kgat/ \
--test_path ../data/bert_test_triplet.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/triplet_kgat/model.best.pt \
--name test.json \
--classifier kgat \
--gpu 0

python fever_score_test.py --predicted_labels ./output/triplet_kgat/dev.json  --predicted_evidence ../data/bert_eval_triplet.json --actual ../data/dev_eval.json
#python $VSC_DATA/projects/KernelGAT/kgat/fever_score_test.py --predicted_labels $VSC_DATA/projects/KernelGAT/output/kgat/npretrain/test.json  --predicted_evidence $VSC_DATA/projects/KernelGAT/data/bert_test.json --actual ../data/test_eval.json
