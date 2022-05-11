python preprocess.py --data_dir ../GANRanker/data/glge_new/xsum --dataset_name xsum \
    --candidate_dir /weizhou_data/generation/bart-samsum/bart-universal-generation/results/xsum-large \
    --num_cand 16 --save_name xsum-large --tokenizer_type roberta --tokenizer_dir /weizhou_data/models/roberta-large


# if process bart data
# python preprocess.py --data_dir ../GANRanker/data/glge_new/xsum --dataset_name xsum \
#     --candidate_dir /weizhou_data/generation/bart-samsum/bart-universal-generation/results/xsum-large \
#     --num_cand 16 --save_name xsum-large-bart --tokenizer_type bart --tokenizer_dir /weizhou_data/models/bart-large