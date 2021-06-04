python train_pointwise.py --outdir ../checkpoint/pretrain/pointwise_pretrain_evi_num_5_slate_20/ \
--train_path ../data/all_train.json \
--valid_path ../data/all_dev.json \
--bert_pretrain ../bert_base \
--num_train_epochs 2.0 \
--slate_length 20 \
--gpu 0

python test.py --outdir ./output/ \
--test_path ../data/all_dev.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/pretrain/pointwise_pretrain_evi_num_5_slate_20/model.best.pt \
--name dev.json \
--gpu 0


python test.py --outdir ./output/ \
--test_path ../data/all_test.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/pretrain/pointwise_pretrain_evi_num_5_slate_20/model.best.pt \
--name test.json \
--gpu 0

python test.py --outdir ./output/ \
--test_path ../data/all_train.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/pretrain/pointwise_pretrain_evi_num_5_slate_20/model.best.pt \
--name train.json \
--gpu 0

python process_data.py --retrieval_file ./output/dev.json --gold_file ../data/golden_dev.json --output ../data/bert_dev_pointwise_pretrain_evi_num_5_slate_20.json
python process_data.py --retrieval_file ./output/dev.json --gold_file ../data/golden_dev.json --output ../data/bert_eval_pointwise_pretrain_evi_num_5_slate_20.json --test
python process_data.py --retrieval_file ./output/test.json --gold_file ../data/all_test.json  --output ../data/bert_test_pointwise_pretrain_evi_num_5_slate_20.json --test
python process_data.py --retrieval_file ./output/train.json --gold_file ../data/golden_train.json --output ../data/bert_train_pointwise_pretrain_evi_num_5_slate_20.json