python train.py --outdir ../checkpoint/angular_kgat/ \
--train_path ../data/bert_train_angular.json \
--valid_path ../data/bert_dev_angular.json \
--bert_pretrain ../bert_base \
--postpretrain ../pretrain/save_model/model.best.pt \
--train_batch_size 6 \
--classifier kgat \
--gpu 0

python test.py --outdir ./output/angular_kgat/ \
--test_path ../data/bert_eval_angular.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/angular_kgat/model.best.pt \
--name dev.json \
--classifier kgat \
--gpu 0

python test.py --outdir ./output/angular_kgat/ \
--test_path ../data/bert_test_angular.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/angular_kgat/model.best.pt \
--name test.json \
--classifier kgat \
--gpu 0

python fever_score_test.py --predicted_labels ./output/angular_kgat/dev.json  --predicted_evidence ../data/bert_eval_angular.json --actual ../data/dev_eval.json
#python $VSC_DATA/projects/KernelGAT/kgat/fever_score_test.py --predicted_labels $VSC_DATA/projects/KernelGAT/output/kgat/npretrain/test.json  --predicted_evidence $VSC_DATA/projects/KernelGAT/data/bert_test.json --actual ../data/test_eval.json
