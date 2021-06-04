python train.py --outdir ../checkpoint/triplet_model \
--train_path ../data/train_pair \
--valid_path ../data/dev_pair \
--bert_pretrain ../bert_base \
--train_batch_size 8 \
--eval_step 2000

python test.py --outdir ./output/ \
--test_path ../data/all_dev.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/triplet_model_tanh/model.best.pt \
--name dev.json \
--batch_size 16 \
--gpu 0


python test.py --outdir ./output/ \
--test_path ../data/all_test.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/triplet_model_tanh/model.best.pt \
--name test.json \
--batch_size 16 \
--gpu 0

python test.py --outdir ./output/ \
--test_path ../data/all_train.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/triplet_model_tanh/model.best.pt \
--name train.json \
--batch_size 16 \
--gpu 0

python process_data.py --retrieval_file ./output/dev.json --gold_file ../data/golden_dev.json --output ../data/bert_dev_triplet.json
python process_data.py --retrieval_file ./output/dev.json --gold_file ../data/golden_dev.json --output ../data/bert_eval_triplet.json --test
python process_data.py --retrieval_file ./output/test.json --gold_file ../data/all_test.json  --output ../data/bert_test_triplet.json --test
python process_data.py --retrieval_file ./output/train.json --gold_file ../data/golden_train.json --output ../data/bert_train_triplet.json