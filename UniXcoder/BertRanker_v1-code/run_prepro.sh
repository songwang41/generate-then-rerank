python preprocess.py --data_dir /weizhou_data/generation/UniXcoder/code-generation/data/concode/train.json \
    --dataset_name concode \
    --candidate_dir /weizhou_data/generation/UniXcoder/code-generation/predictions/train/gen_from_16/predictions.txt \
    --num_cand 16 --save_dir data/concode/train_debug \
    --tokenizer_type roberta --tokenizer_dir /weizhou_data/models/unixcoder-base


python preprocess.py --data_dir /weizhou_data/generation/UniXcoder/code-generation/data/concode/dev.json \
    --dataset_name concode \
    --candidate_dir /weizhou_data/generation/UniXcoder/code-generation/predictions/train/gen_from_16/predictions.txt \
    --num_cand 16 --save_dir data/concode/dev_debug \
    --tokenizer_type roberta --tokenizer_dir /weizhou_data/models/unixcoder-base


python preprocess.py --data_dir /weizhou_data/generation/UniXcoder/code-generation/data/concode/test.json \
    --dataset_name concode \
    --candidate_dir /weizhou_data/generation/UniXcoder/code-generation/predictions/train/gen_from_16/predictions.txt \
    --num_cand 16 --save_dir data/concode/test_debug \
    --tokenizer_type roberta --tokenizer_dir /weizhou_data/models/unixcoder-base