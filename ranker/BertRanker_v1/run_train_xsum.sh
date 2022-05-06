python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_reranker.py --overwrite_output_dir \
    --task_name sum --dataset_name xsum \
    --train_data_path xsum-large \
    --dev_data_path xsum-large \
    --test_data_path xsum-large \
    --do_train True --do_eval True --do_predict True --prediction_loss_only False \
    --use_untokenized_data True \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
    --num_train_epochs 6 \
    --evaluation_strategy steps --eval_steps 750 \
    --logging_strategy steps --logging_steps 750 \
    --save_strategy steps --save_steps 750 --save_total_limit 10 \
    --load_best_model_at_end True \
    --metric_for_best_model rouge2 --greater_is_better True \
    --model_name_or_path roberta-large \
    --output_dir saves/roberta-large-xsum \
    --num_cand 16 --max_num 3 \
    --loss_type contrastive \
    --max_source_length 400 --max_candidate_length 109 --position_extend_way normal \
    --cache_data