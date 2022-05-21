python -m torch.distributed.launch --nproc_per_node 8 --master_port 6009 run_train.py --overwrite_output_dir \
    --task_name sum --dataset_name cnndm \
    --train_data_path /weizhou_data/generation/ranker/GANRanker/data/cnndm \
    --dev_data_path /weizhou_data/generation/ranker/GANRanker/data/cnndm \
    --test_data_path /weizhou_data/generation/ranker/GANRanker/data/cnndm \
    --load_tokenized_data True \
    --generator_supervised True --generator_supervised_lambda 1.0 --reward_scaler 1.5 \
    --evaluate_generator True --generate_eval_candidates True \
    --use_baseline_reward True --reward_type reranker --training_mode iterative \
    --num_cand_generated 16 --num_cand_picked 3 --candidate_pick_strategy bottom \
    --loss_type contrastive \
    --do_train True --do_eval True --do_predict False --prediction_loss_only False \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --evaluation_strategy steps --eval_steps 2000 \
    --logging_strategy steps --logging_steps 1000 \
    --save_strategy steps --save_steps 2000 --save_total_limit 10 \
    --iteration_steps 2000 --iteration_reranker_steps 1000 \
    --load_best_model_at_end True \
    --metric_for_best_model reranker_eval_rouge1 --greater_is_better True \
    --reranker_model_name_or_path /weizhou_data/models/roberta-base \
    --generator_model_name_or_path /weizhou_data/models/bart-base \
    --output_dir saves/rl_ranker_v1_all_data_no_warmup \
    --generator_max_source_length 1020 --reranker_max_source_length 400 --max_target_length 100 \
    --position_extend_way copy \
    --cache_data \
    --disable_tqdm False \
