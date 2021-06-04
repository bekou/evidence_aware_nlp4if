python train.py --outdir ../checkpoint/retrieval_model_evi_num_10_slate_20 \
--train_path ../data/train_pair_evi_num_10_slate_20 \
--valid_path ../data/dev_pair_evi_num_10_slate_20 \
--bert_pretrain ../bert_base \
--train_batch_size 16 \
--early_stop 8 \
--eval_step 2000 \
--gpu 0

python test.py --outdir ./output/evi_num_10_slate_20/ \
--test_path ../data/all_dev.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/retrieval_model_evi_num_10_slate_20/model.best.pt \
--name dev.json \
--gpu 0

python test.py --outdir ./output/evi_num_10_slate_20/ \
--test_path ../data/all_test.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/retrieval_model_evi_num_10_slate_20/model.best.pt \
--name test.json \
--gpu 0

python test.py --outdir ./output/evi_num_10_slate_20/ \
--test_path ../data/all_train.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/retrieval_model_evi_num_10_slate_20/model.best.pt \
--name train.json \
--gpu 0


python process_data.py --retrieval_file ./output/evi_num_10_slate_20/train.json --gold_file ../data/golden_train.json --output ../data/bert_train_pw_evi_num_10_slate_20.json
python process_data.py --retrieval_file ./output/evi_num_10_slate_20/dev.json --gold_file ../data/golden_dev.json --output ../data/bert_dev_pw_evi_num_10_slate_20.json
python process_data.py --retrieval_file ./output/evi_num_10_slate_20/dev.json --gold_file ../data/golden_dev.json --output ../data/bert_eval_pw_evi_num_10_slate_20.json --test
python process_data.py --retrieval_file ./output/evi_num_10_slate_20/test.json --gold_file ../data/all_test.json  --output ../data/bert_test_pw_evi_num_10_slate_20.json --test