# first half
python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_summarization_half.py  --overwrite_output_dir \
--dataset_name cnndm --half_name train_1 \
--do_train --do_eval --do_predict --evaluation_strategy epoch --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --num_train_epochs 5 --logging_strategy epoch --save_strategy epoch --save_total_limit 10  \
--load_best_model_at_end True --metric_for_best_model rouge2 --dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
--generation_num_beams 4  \
--model_name_or_path /weizhou_data/models/bart-base/pytorch_model.bin \
--config_name /weizhou_data/models/bart-base/config.json \
--tokenizer_name /weizhou_data/models/bart-base/ \
--max_source_length 1000 --max_target_length 100 --disable_tqdm False \
--output_dir saves/cnndm_half/half_1


# second half
python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_summarization_half.py  --overwrite_output_dir \
--dataset_name cnndm --half_name train_2 \
--do_train --do_eval --do_predict --evaluation_strategy epoch --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --num_train_epochs 5 --logging_strategy epoch --save_strategy epoch --save_total_limit 10  \
--load_best_model_at_end True --metric_for_best_model rouge2 --dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
--generation_num_beams 4  \
--model_name_or_path /weizhou_data/models/bart-base/pytorch_model.bin \
--config_name /weizhou_data/models/bart-base/config.json \
--tokenizer_name /weizhou_data/models/bart-base/ \
--max_source_length 1000 --max_target_length 100 --disable_tqdm False \
--output_dir saves/cnndm_half/half_2