# test runing
for i in {1..40} 
do
    python main.py --cuda --gpuid 0 -e \
    --log_name samsum-from32-noaug \
    --model_name_or_path /weizhou_data/models/roberta-base \
    --state_dict_path samsum-from32-noaug/scorer.bin \
    --cache_data \
    --eval_batch_size 8 \
    --dataset samsum \
    --max_num $i \
    --num_beams $i
done