python -m torch.distributed.launch --nproc_per_node 8 --master_port 6009 run_train.py --overwrite_output_dir \
    --task_name sum --dataset_name glge_new/xsum \
    --train_data_path /weizhou_data/generation/ranker/GANRanker/data/glge_new/xsum \
    --dev_data_path /weizhou_data/generation/ranker/GANRanker/data/glge_new/xsum \
    --test_data_path /weizhou_data/generation/ranker/GANRanker/data/glge_new/xsum \
    --load_tokenized_data False \
    --generator_supervised True --generator_supervised_lambda 1.0 --reward_scaler 1.5 \
    --evaluate_generator True --generate_eval_candidates True \
    --use_baseline_reward True --reward_type reranker --training_mode iterative \
    --num_cand_generated 16 --num_cand_picked 3 --candidate_pick_strategy bottom \
    --loss_type contrastive \
    --do_train True --do_eval True --do_predict False --prediction_loss_only False \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 6 \
    --evaluation_strategy steps --eval_steps 1500 \
    --logging_strategy steps --logging_steps 750 \
    --save_strategy steps --save_steps 1500 --save_total_limit 20 \
    --iteration_steps 1500 --iteration_reranker_steps 750 \
    --load_best_model_at_end True \
    --metric_for_best_model reranker_eval_rouge2 --greater_is_better True \
    --reranker_model_name_or_path /weizhou_data/generation/ranker/BertRanker_v1/saves/roberta-large-xsum \
    --generator_model_name_or_path /weizhou_data/generation/bart-samsum/bart-universal-generation/saves/bart-large-xsum \
    --output_dir saves/rl_ranker_v4_large_xsum \
    --generator_max_source_length 1020 --reranker_max_source_length 400 --max_target_length 109 \
    --position_extend_way copy \
    --cache_data \
    --disable_tqdm False \
