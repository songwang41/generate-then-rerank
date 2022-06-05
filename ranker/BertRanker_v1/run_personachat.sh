# for debug
# CUDA_VISIBLE_DEVICES="" 
# python  run_reranker.py --overwrite_output_dir \
#     --task_name dialog --dataset_name personachat \
#     --report_to none \
#     --train_data_path data/personachat-large-new/train/gen_from_16 \
#     --dev_data_path data/personachat-large-new/dev/gen_from_16 \
#     --test_data_path data/personachat-large-new/test/gen_from_16 \
#     --do_train True --do_eval True --do_predict True --prediction_loss_only False \
#     --use_untokenized_data False \
#     --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 1e-5 --warmup_ratio 0.2 \
#     --num_train_epochs 3 \
#     --evaluation_strategy steps --eval_steps 500 \
#     --logging_strategy steps --logging_steps 500 \
#     --save_strategy steps --save_steps 500 --save_total_limit 20 \
#     --load_best_model_at_end True \
#     --metric_for_best_model bleu_1 --greater_is_better True \
#     --model_name_or_path /weizhou_data/models/roberta-large \
#     --output_dir saves/roberta-large-personachat-new-lr1e-5-wu02 \
#     --num_cand 16 --max_num 3 \
#     --loss_type contrastive \
#     --max_source_length 430 --max_candidate_length 70 --position_extend_way normal \
#     --cache_data --disable_tqdm True \

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_reranker.py --overwrite_output_dir \
    --task_name dialog --dataset_name personachat \
    --report_to none \
    --train_data_path data/personachat-large-new/train/gen_from_16 \
    --dev_data_path data/personachat-large-new/dev/gen_from_16 \
    --test_data_path data/personachat-large-new/test/gen_from_16 \
    --do_train True --do_eval True --do_predict True --prediction_loss_only False \
    --use_untokenized_data False \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 --warmup_ratio 0.2 \
    --num_train_epochs 3 \
    --evaluation_strategy steps --eval_steps 500 \
    --logging_strategy steps --logging_steps 500 \
    --save_strategy steps --save_steps 500 --save_total_limit 20 \
    --load_best_model_at_end True \
    --metric_for_best_model bleu_1 --greater_is_better True \
    --model_name_or_path /weizhou_data/models/roberta-large \
    --output_dir saves/roberta-large-personachat-new-lr1e-5-wu02 \
    --num_cand 16 --max_num 3 \
    --loss_type contrastive \
    --max_source_length 430 --max_candidate_length 70 --position_extend_way normal \
    --cache_data --disable_tqdm True \


python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_reranker.py --overwrite_output_dir \
    --task_name dialog --dataset_name personachat \
    --report_to none \
    --train_data_path data/personachat-large-new/train/gen_from_16 \
    --dev_data_path data/personachat-large-new/dev/gen_from_16 \
    --test_data_path data/personachat-large-new/test/gen_from_16 \
    --do_train True --do_eval True --do_predict True --prediction_loss_only False \
    --use_untokenized_data False \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 --warmup_ratio 0.3 \
    --num_train_epochs 3 \
    --evaluation_strategy steps --eval_steps 500 \
    --logging_strategy steps --logging_steps 500 \
    --save_strategy steps --save_steps 500 --save_total_limit 20 \
    --load_best_model_at_end True \
    --metric_for_best_model bleu_1 --greater_is_better True \
    --model_name_or_path /weizhou_data/models/roberta-large \
    --output_dir saves/roberta-large-personachat-new-lr1e-5-wu03 \
    --num_cand 16 --max_num 3 \
    --loss_type contrastive \
    --max_source_length 430 --max_candidate_length 70 --position_extend_way normal \
    --cache_data --disable_tqdm True \


python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_reranker.py --overwrite_output_dir \
    --task_name dialog --dataset_name personachat \
    --report_to none \
    --train_data_path data/personachat-large-new/train/gen_from_16 \
    --dev_data_path data/personachat-large-new/dev/gen_from_16 \
    --test_data_path data/personachat-large-new/test/gen_from_16 \
    --do_train True --do_eval True --do_predict True --prediction_loss_only False \
    --use_untokenized_data False \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 --warmup_ratio 0.4 \
    --num_train_epochs 3 \
    --evaluation_strategy steps --eval_steps 500 \
    --logging_strategy steps --logging_steps 500 \
    --save_strategy steps --save_steps 500 --save_total_limit 20 \
    --load_best_model_at_end True \
    --metric_for_best_model bleu_1 --greater_is_better True \
    --model_name_or_path /weizhou_data/models/roberta-large \
    --output_dir saves/roberta-large-personachat-new-lr1e-5-wu04 \
    --num_cand 16 --max_num 3 \
    --loss_type contrastive \
    --max_source_length 430 --max_candidate_length 70 --position_extend_way normal \
    --cache_data --disable_tqdm True \


python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_reranker.py --overwrite_output_dir \
    --task_name dialog --dataset_name personachat \
    --report_to none \
    --train_data_path data/personachat-large-new/train/gen_from_16 \
    --dev_data_path data/personachat-large-new/dev/gen_from_16 \
    --test_data_path data/personachat-large-new/test/gen_from_16 \
    --do_train True --do_eval True --do_predict True --prediction_loss_only False \
    --use_untokenized_data False \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 --warmup_ratio 0.3 \
    --num_train_epochs 1000 \
    --evaluation_strategy steps --eval_steps 500 \
    --logging_strategy steps --logging_steps 500 \
    --save_strategy steps --save_steps 500 --save_total_limit 20 \
    --load_best_model_at_end True \
    --metric_for_best_model bleu_1 --greater_is_better True \
    --model_name_or_path /weizhou_data/models/roberta-large \
    --output_dir saves/roberta-large-personachat-save_gpu \
    --num_cand 16 --max_num 3 \
    --loss_type contrastive \
    --max_source_length 430 --max_candidate_length 70 --position_extend_way normal \
    --cache_data --disable_tqdm True \