#python train_pointwise.py --outdir ../checkpoint/cosine_classifier/ \
#--train_path ../data/all_train.json \
#--valid_path ../data/all_dev.json \
#--bert_pretrain ../bert_base \
#--num_train_epochs 2.0 \
#--warmup_proportion 0.1

python3 train.py --outdir ../checkpoint/cosine_classifier \
--train_path ../data/train_pair \
--valid_path ../data/dev_pair \
--bert_pretrain ../bert_base \
--train_batch_size 16 \
--eval_step 2000 \
--dropout 0.6 \
--gpu 0

python3 test.py --outdir ./output/ \
--test_path ../data/all_dev.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/cosine_classifier/model.best.pt \
--name dev.json \
--batch_size 16 \
--gpu 0

python3 test.py --outdir ./output/ \
--test_path ../data/all_test.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/cosine_classifier/model.best.pt \
--name test.json \
--batch_size 16 \
--gpu 0

python3 test.py --outdir ./output/ \
--test_path ../data/all_train.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/cosine_classifier/model.best.pt \
--name train.json \
--batch_size 16 \
--gpu 0

python3 process_data.py --retrieval_file ./output/dev.json --gold_file ../data/golden_dev.json --output ../data/bert_dev_cosine.json
python3 process_data.py --retrieval_file ./output/dev.json --gold_file ../data/golden_dev.json --output ../data/bert_eval_cosine.json --test
python3 process_data.py --retrieval_file ./output/test.json --gold_file ../data/all_test.json  --output ../data/bert_test_cosine.json --test
python3 process_data.py --retrieval_file ./output/train.json --gold_file ../data/golden_train.json --output ../data/bert_train_cosine.json
