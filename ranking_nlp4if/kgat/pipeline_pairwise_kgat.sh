python train.py --outdir ../checkpoint/retrieval_kgat/ \
--train_path ../data/bert_train.json \
--valid_path ../data/bert_dev.json \
--bert_pretrain ../bert_base \
--train_batch_size 4 \
--classifier kgat \
--gpu 1

python test.py --outdir ./output/retrieval_kgat/ \
--test_path ../data/bert_eval.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/retrieval_kgat/model.best.pt \
--name dev.json \
--classifier kgat \
--gpu 1

python test.py --outdir ./output/retrieval_kgat/ \
--test_path ../data/bert_test.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/retrieval_kgat/model.best.pt \
--name test.json \
--classifier kgat \
--gpu 1

python fever_score_test.py --predicted_labels ./output/retrieval_kgat/dev.json  --predicted_evidence ../data/bert_eval.json --actual ../data/dev_eval.json
#python $VSC_DATA/projects/KernelGAT/kgat/fever_score_test.py --predicted_labels $VSC_DATA/projects/KernelGAT/output/kgat/npretrain/test.json  --predicted_evidence $VSC_DATA/projects/KernelGAT/data/bert_test.json --actual ../data/test_eval.json
