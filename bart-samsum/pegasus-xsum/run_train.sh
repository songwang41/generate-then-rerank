# DEBUG
# CUDA_VISIBLE_DEVICES="" python  run_train.py  --overwrite_output_dir \
# --dataset_name xsum --use_tokenized_data False \
# --report_to none \
# --do_train True --do_eval True --do_predict True --evaluation_strategy epoch --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --gradient_accumulation_steps 1 \
# --learning_rate 5e-5 --num_train_epochs 5 --logging_strategy epoch --save_strategy epoch --save_total_limit 10  \
# --load_best_model_at_end True --metric_for_best_model rouge2 --dataloader_pin_memory True --predict_with_generate True --generation_max_length 109 \
# --generation_num_beams 4 --generation_max_length 109 \
# --model_name_or_path /weizhou_data/models/pegasus-large \
# --config_name /weizhou_data/models/pegasus-large \
# --tokenizer_name /weizhou_data/models/pegasus-large \
# --max_source_length 1020 --max_target_length 109 --disable_tqdm False \
# --output_dir saves/pegasus-large-xsum

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_train.py  --overwrite_output_dir \
--dataset_name xsum --use_tokenized_data False \
--report_to none \
--do_train True --do_eval True --do_predict True --evaluation_strategy epoch --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --num_train_epochs 5 --logging_strategy epoch --save_strategy epoch --save_total_limit 10  \
--load_best_model_at_end True --metric_for_best_model rouge2 --dataloader_pin_memory True --predict_with_generate True --generation_max_length 109 \
--generation_num_beams 4 --generation_max_length 109 \
--model_name_or_path /weizhou_data/models/pegasus-large \
--config_name /weizhou_data/models/pegasus-large \
--tokenizer_name /weizhou_data/models/pegasus-large \
--max_source_length 1020 --max_target_length 109 --disable_tqdm False \
--output_dir saves/pegasus-large-xsum

# predict with pegasus-xsum
CUDA_VISIBLE_DEVICES="" python run_train.py  --overwrite_output_dir \
--dataset_name xsum --use_tokenized_data False \
--report_to none \
--do_train False --do_eval True --do_predict True --evaluation_strategy epoch --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --num_train_epochs 5 --logging_strategy epoch --save_strategy epoch --save_total_limit 10  \
--load_best_model_at_end True --metric_for_best_model rouge2 --dataloader_pin_memory True --predict_with_generate True --generation_max_length 109 \
--generation_num_beams 4 --generation_max_length 109 \
--model_name_or_path /weizhou_data/models/pegasus-xsum \
--config_name /weizhou_data/models/pegasus-xsum \
--tokenizer_name /weizhou_data/models/pegasus-xsum \
--max_source_length 510 --max_target_length 109 --disable_tqdm False \
--output_dir saves/pegasus-xsum

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_train.py  --overwrite_output_dir \
--dataset_name xsum --use_tokenized_data False \
--report_to none \
--do_train False --do_eval False --do_predict True --evaluation_strategy no --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --num_train_epochs 5 --logging_strategy epoch --save_strategy epoch --save_total_limit 10  \
--load_best_model_at_end False --metric_for_best_model rouge2 --dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
--generation_num_beams 8 --generation_max_length 100 \
--model_name_or_path /weizhou_data/models/pegasus-xsum \
--config_name /weizhou_data/models/pegasus-xsum \
--tokenizer_name /weizhou_data/models/pegasus-xsum \
--max_source_length 510 --max_target_length 100 --disable_tqdm False \
--output_dir saves/pegasus-xsum