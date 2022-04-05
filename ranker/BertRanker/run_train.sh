# python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_reranker.py --overwrite_output_dir \
#     --task_name sum --dataset_name cnndm_debug \
#     --do_train --do_eval --do_predict --prediction_loss_only False \
#     --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
#     --num_train_epochs 3 \
#     --evaluation_strategy steps --eval_steps 20 \
#     --logging_strategy steps --logging_steps 20 \
#     --save_strategy no --save_steps 40 --save_total_limit 10 \
#     --load_best_model_at_end False \
#     --metric_for_best_model rouge1 --greater_is_better True \
#     --model_name_or_path /weizhou_data/models/roberta-base \
#     --output_dir saves/debug \
#     --num_cand 16 --max_num 2 \
#     --loss_type binary \
#     --cache_data \


python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_reranker.py --overwrite_output_dir \
    --task_name sum --dataset_name cnndm_debug \
    --train_data_path data/cnndm_debug/train_half/gen_from_16 \
    --test_data_path data/cnndm_debug/test/gen_from_16 \
    --do_train True --do_eval False --do_predict True --prediction_loss_only False \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --evaluation_strategy no --eval_steps 20 \
    --logging_strategy no --logging_steps 20 \
    --save_strategy no --save_steps 40 --save_total_limit 10 \
    --load_best_model_at_end False \
    --metric_for_best_model rouge1 --greater_is_better True \
    --model_name_or_path /weizhou_data/models/roberta-base \
    --output_dir saves/debug \
    --num_cand 16 --max_num 3 \
    --loss_type binary \
    --max_source_length 400 --max_candidate_length 100 --position_extend_way normal \
    --cache_data \

