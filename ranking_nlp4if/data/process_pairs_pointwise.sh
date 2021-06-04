python generate_pair_pointwise.py --infile all_dev.json --outfile dev_pair_evi_num_5 --evi_num 5 --shuffle 1
python generate_pair_pointwise.py --infile all_train.json --outfile train_pair_evi_num_5 --evi_num 5 --shuffle 1

python generate_pair_pointwise.py --infile all_dev.json --outfile dev_pair_evi_num_10 --evi_num 10 --shuffle 1
python generate_pair_pointwise.py --infile all_train.json --outfile train_pair_evi_num_10 --evi_num 10 --shuffle 1

python generate_pair_pointwise.py --infile all_dev.json --outfile dev_pair_evi_num_5_slate_20 --slate_length 20 --evi_num 5 --shuffle 1
python generate_pair_pointwise.py --infile all_train.json --outfile train_pair_evi_num_5_slate_20 --slate_length 20 --evi_num 5 --shuffle 1

python generate_pair_pointwise.py --infile all_dev.json --outfile dev_pair_evi_num_10_slate_20 --slate_length 20 --evi_num 10 --shuffle 1
python generate_pair_pointwise.py --infile all_train.json --outfile train_pair_evi_num_10_slate_20 --slate_length 20 --evi_num 10 --shuffle 1