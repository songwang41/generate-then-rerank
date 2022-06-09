# for debug
# CUDA_VISIBLE_DEVICES="" python run_train.py --overwrite_output_dir \
#     --task_name sum --dataset_name glge_new/xsum \
#     --train_data_path /weizhou_data/generation/ranker/GANRanker/data/glge_new/xsum \
#     --dev_data_path /weizhou_data/generation/ranker/GANRanker/data/glge_new/xsum \
#     --test_data_path /weizhou_data/generation/ranker/GANRanker/data/glge_new/xsum \
#     --load_tokenized_data False \
#     --generator_supervised True --generator_supervised_lambda 1.0 \
#     --reranker_reward_scaler 1.5 --metric_reward_scaler 1.5 \
#     --evaluate_generator True --generate_eval_candidates True \
#     --use_baseline_reward True --reward_type reranker --training_mode iterative \
#     --generator_num_cand_generated 8 --generator_num_cand_picked 8 \
#     --num_cand_generated 16 --num_cand_picked 3 --candidate_pick_strategy bottom \
#     --loss_type contrastive \
#     --do_train True --do_eval True --do_predict False --prediction_loss_only False \
#     --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 6 \
#     --evaluation_strategy steps --eval_steps 1500 \
#     --logging_strategy steps --logging_steps 750 \
#     --save_strategy steps --save_steps 1500 --save_total_limit 20 \
#     --iteration_steps 1500 --iteration_reranker_steps 0 \
#     --load_best_model_at_end True \
#     --metric_for_best_model reranker_eval_rouge2 --greater_is_better True \
#     --reranker_model_name_or_path /weizhou_data/generation/ranker/BertRanker_v1/saves/roberta-large-xsum-song \
#     --generator_model_name_or_path /weizhou_data/models/pegasus-xsum \
#     --output_dir saves/v7-large-pegasus \
#     --generator_max_source_length 510 --reranker_max_source_length 400 --generator_max_target_length 64 --reranker_max_target_length 64 \
#     --position_extend_way copy \
#     --cache_data \
#     --disable_tqdm False \

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6009 run_train.py --overwrite_output_dir \
    --task_name sum --dataset_name glge_new/xsum \
    --train_data_path /weizhou_data/generation/ranker/GANRanker/data/glge_new/xsum \
    --dev_data_path /weizhou_data/generation/ranker/GANRanker/data/glge_new/xsum \
    --test_data_path /weizhou_data/generation/ranker/GANRanker/data/glge_new/xsum \
    --load_tokenized_data False \
    --generator_supervised True --generator_supervised_lambda 1.0 \
    --reranker_reward_scaler 1.5 --metric_reward_scaler 1.5 \
    --evaluate_generator True --generate_eval_candidates True \
    --use_baseline_reward True --reward_type reranker --training_mode iterative \
    --generator_num_cand_generated 8 --generator_num_cand_picked 8 \
    --num_cand_generated 16 --num_cand_picked 3 --candidate_pick_strategy bottom \
    --loss_type contrastive \
    --do_train True --do_eval False --do_predict False --prediction_loss_only False \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 4 \
    --generator_learning_rate 5e-5 --reranker_learning_rate 1e-5 --warmup_ratio 0 \
    --evaluation_strategy no --eval_steps 1500 \
    --logging_strategy steps --logging_steps 750 \
    --save_strategy steps --save_steps 1500 --save_total_limit 20 \
    --iteration_steps 1500 --iteration_reranker_steps 750 \
    --load_best_model_at_end True \
    --metric_for_best_model reranker_eval_rouge2 --greater_is_better True \
    --reranker_model_name_or_path /weizhou_data/generation/ranker/BertRanker_v1/saves/roberta-large-xsum-song \
    --generator_model_name_or_path /weizhou_data/models/pegasus-xsum \
    --output_dir saves/v7-large-pegasus \
    --generator_max_source_length 510 --reranker_max_source_length 400 --generator_max_target_length 64 --reranker_max_target_length 64 \
    --position_extend_way copy \
    --cache_data \
    --disable_tqdm True \


beam_size=16
strategy=beamsearch
for ((i=1000; i<=15000; i += 1000))
do
    python -m torch.distributed.launch --nproc_per_node 8 --master_port 6009 run_train.py --overwrite_output_dir \
    --task_name qg --dataset_name glge/squadqg \
    --train_data_path /weizhou_data/generation/ranker/GANRanker/data/glge/squadqg \
    --dev_data_path /weizhou_data/generation/ranker/GANRanker/data/glge/squadqg \
    --test_data_path /weizhou_data/generation/ranker/GANRanker/data/glge/squadqg \
        --load_tokenized_data False \
        --generator_supervised True --generator_supervised_lambda 1.0 \
        --reranker_reward_scaler 1.5 --metric_reward_scaler 1.5 \
        --evaluate_generator True --generate_eval_candidates True --generate_candidate_strategy $strategy \
        --use_baseline_reward True --training_mode iterative \
        --generator_num_cand_generated $beam_size --generator_num_cand_picked $beam_size \
        --num_cand_generated 16 --num_cand_picked 3 --candidate_pick_strategy bottom \
        --loss_type contrastive \
        --do_train False --do_eval True --do_predict True --prediction_loss_only False \
        --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --generator_learning_rate 5e-5 --reranker_learning_rate 1e-5 --warmup_ratio 0 \
        --num_train_epochs 3 \
        --evaluation_strategy steps --eval_steps 1000 \
        --logging_strategy steps --logging_steps 1000 \
        --save_strategy steps --save_steps 1000 --save_total_limit 10 \
        --iteration_steps 2000 --iteration_reranker_steps 1000 \
        --load_best_model_at_end True \
        --metric_for_best_model generator_eval_rouge1 --greater_is_better True \
        --reranker_model_name_or_path saves/v7-large-pegasus/checkpoint-$i/reranker \
        --generator_model_name_or_path saves/v7-large-pegasus/checkpoint-$i/generator \
        --output_dir saves/v7-large-pegasus/checkpoint-$i/$strategy-$beam_size \
        --generator_max_source_length 510 --reranker_max_source_length 400 --generator_max_target_length 64 --reranker_max_target_length 64 \
        --cache_data \
        --disable_tqdm True
done

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6009 run_train.py --overwrite_output_dir \
    --task_name sum --dataset_name glge_new/xsum \
    --train_data_path /weizhou_data/generation/ranker/GANRanker/data/glge_new/xsum \
    --dev_data_path /weizhou_data/generation/ranker/GANRanker/data/glge_new/xsum \
    --test_data_path /weizhou_data/generation/ranker/GANRanker/data/glge_new/xsum \
    --load_tokenized_data False \
    --generator_supervised True --generator_supervised_lambda 1.0 \
    --reranker_reward_scaler 1.5 --metric_reward_scaler 1.5 \
    --evaluate_generator True --generate_eval_candidates True \
    --use_baseline_reward True --reward_type reranker --training_mode iterative \
    --generator_num_cand_generated 8 --generator_num_cand_picked 8 \
    --num_cand_generated 16 --num_cand_picked 3 --candidate_pick_strategy bottom \
    --loss_type contrastive \
    --do_train True --do_eval True --do_predict False --prediction_loss_only False \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2000 \
    --generator_learning_rate 5e-5 --reranker_learning_rate 1e-5 --warmup_ratio 0 \
    --evaluation_strategy steps --eval_steps 1500 \
    --logging_strategy steps --logging_steps 750 \
    --save_strategy steps --save_steps 1500 --save_total_limit 20 \
    --iteration_steps 1500 --iteration_reranker_steps 750 \
    --load_best_model_at_end True \
    --metric_for_best_model reranker_eval_rouge2 --greater_is_better True \
    --reranker_model_name_or_path /weizhou_data/generation/ranker/BertRanker_v1/saves/roberta-large-xsum-song \
    --generator_model_name_or_path /weizhou_data/models/pegasus-xsum \
    --output_dir saves/v7-large-pegasus-save_gpu \
    --generator_max_source_length 510 --reranker_max_source_length 400 --generator_max_target_length 64 --reranker_max_target_length 64 \
    --position_extend_way copy \
    --cache_data \
    --disable_tqdm True \

