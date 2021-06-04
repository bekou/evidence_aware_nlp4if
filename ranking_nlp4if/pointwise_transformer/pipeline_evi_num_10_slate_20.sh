python train_pointwise.py --outdir ../checkpoint/pretrain/pretrain_transformer_evi_num_10_slate_20/ \
--train_path ../data/all_train.json \
--valid_path ../data/all_dev.json \
--bert_pretrain ../bert_base \
--evi_num 10 \
--slate_length 20 \
--gpu 0

python test.py --outdir ./output/transformer_evi_num_10_slate_20/ \
--test_path ../data/all_dev.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/pretrain/pretrain_transformer_evi_num_10_slate_20/model.best.pt \
--name dev.json \
--gpu 0

python test.py --outdir ./output/transformer_evi_num_10_slate_20/ \
--test_path ../data/all_test.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/pretrain/pretrain_transformer_evi_num_10_slate_20/model.best.pt \
--name test.json \
--gpu 0

python test.py --outdir ./output/transformer_evi_num_10_slate_20/ \
--test_path ../data/all_train.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/pretrain/pretrain_transformer_evi_num_10_slate_20/model.best.pt \
--name train.json \
--gpu 0

python process_data.py --retrieval_file ./output/transformer_evi_num_10_slate_20/dev.json --gold_file ../data/golden_dev.json --output ../data/bert_dev_transformer_evi_num_10_slate_20.json
python process_data.py --retrieval_file ./output/transformer_evi_num_10_slate_20/dev.json --gold_file ../data/golden_dev.json --output ../data/bert_eval_transformer_evi_num_10_slate_20.json --test
python process_data.py --retrieval_file ./output/transformer_evi_num_10_slate_20/test.json --gold_file ../data/all_test.json  --output ../data/bert_test_transformer_evi_num_10_slate_20.json --test
python process_data.py --retrieval_file ./output/transformer_evi_num_10_slate_20/train.json --gold_file ../data/golden_train.json --output ../data/bert_train_transformer_evi_num_10_slate_20.json
