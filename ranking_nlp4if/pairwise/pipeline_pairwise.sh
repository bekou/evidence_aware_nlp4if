python train.py --outdir ../checkpoint/retrieval_model \
--train_path ../data/train_pair \
--valid_path ../data/dev_pair \
--bert_pretrain ../bert_base \
--train_batch_size 16 \
--early_stop 8 \
--eval_step 2000 \
--gpu 0

python test.py --outdir ./output/ \
--test_path ../data/all_dev.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/retrieval_model/model.best.pt \
--name dev.json \
--gpu 0

python test.py --outdir ./output/ \
--test_path ../data/all_test.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/retrieval_model/model.best.pt \
--name test.json \
--gpu 0

python test.py --outdir ./output/ \
--test_path ../data/all_train.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/retrieval_model/model.best.pt \
--name train.json \
--gpu 0


python process_data.py --retrieval_file ./output/train.json --gold_file ../data/golden_train.json --output ../data/bert_train.json
python process_data.py --retrieval_file ./output/dev.json --gold_file ../data/golden_dev.json --output ../data/bert_dev.json
python process_data.py --retrieval_file ./output/dev.json --gold_file ../data/golden_dev.json --output ../data/bert_eval.json --test
python process_data.py --retrieval_file ./output/test.json --gold_file ../data/all_test.json  --output ../data/bert_test.json --test