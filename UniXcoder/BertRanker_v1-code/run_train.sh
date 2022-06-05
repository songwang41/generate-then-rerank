# for debug
python -m torch.distributed.launch --nproc_per_node 2 --master_port 6006 run_reranker.py --overwrite_output_dir \
    --task_name dialog --dataset_name concode \
    --train_data_path /weizhou_data/generation/UniXcoder/BertRanker_v1-code/data/concode/train_debug \
    --dev_data_path /weizhou_data/generation/UniXcoder/BertRanker_v1-code/data/concode/dev_diverse \
    --test_data_path /weizhou_data/generation/UniXcoder/BertRanker_v1-code/data/concode/test_diverse \
    --do_train True --do_eval True --do_predict True --prediction_loss_only False \
    --use_untokenized_data False \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 --warmup_ratio 0.2 \
    --evaluation_strategy steps --eval_steps 200 \
    --logging_strategy steps --logging_steps 200 \
    --save_strategy steps --save_steps 200 --save_total_limit 10 \
    --load_best_model_at_end True \
    --metric_for_best_model bleu --greater_is_better True \
    --model_name_or_path /weizhou_data/models/unixcoder-base \
    --output_dir saves/concode-debug \
    --num_cand 16 --max_num 3 \
    --loss_type contrastive \
    --max_source_length 350 --max_candidate_length 150 --position_extend_way normal \
    --cache_data 

