python evaluate.py \
    --data_dir /weizhou_data/generation/UniXcoder/code-generation/data/concode \
    --dataset_name concode \
    --prediction_dir /weizhou_data/generation/UniXcoder/code-generation/saved_models/predictions.txt \
    --ref_file_type json 


python evaluate.py \
    --data_dir /weizhou_data/generation/UniXcoder/code-generation/data/concode/test_tgt.txt \
    --dataset_name concode \
    --prediction_dir /weizhou_data/generation/UniXcoder/code-generation/saved_models/predictions.txt \
    --ref_file_type txt 