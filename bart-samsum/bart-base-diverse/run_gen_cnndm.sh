python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
--dataset_name cnndm \
--split train \
--do_predict --evaluation_strategy epoch --per_device_train_batch_size 20 --per_device_eval_batch_size 20 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --num_train_epochs 20 --logging_strategy epoch --save_strategy epoch --save_total_limit 10  \
--load_best_model_at_end True --metric_for_best_model rouge2 --dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
--generation_num_beams  16 \
--eval_accumulation_steps 800 \
--generation_num_return_sequences 16 \
--generation_num_beam_groups 16 \
--generation_diversity_penalty 1 \
--model_name_or_path saves/cnndm/pytorch_model.bin \
--config_name saves/cnndm/config.json \
--tokenizer_name saves/cnndm/ \
--max_source_length 1000 --max_target_length 100 --disable_tqdm True \
--output_dir diverse-test

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
--dataset_name cnndm \
--split train \
--do_predict --evaluation_strategy epoch --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --num_train_epochs 20 --logging_strategy epoch --save_strategy epoch --save_total_limit 10  \
--load_best_model_at_end True --metric_for_best_model rouge2 --dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
--generation_num_beams  32 \
--eval_accumulation_steps 800 \
--generation_num_return_sequences 32 \
--generation_num_beam_groups 32 \
--generation_diversity_penalty 1 \
--model_name_or_path saves/cnndm/pytorch_model.bin \
--config_name saves/cnndm/config.json \
--tokenizer_name saves/cnndm/ \
--max_source_length 1000 --max_target_length 100 --disable_tqdm True \
--output_dir diverse-test


# python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
# --dataset_name cnndm \
# --split dev \
# --do_predict --evaluation_strategy epoch --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 \
# --learning_rate 5e-5 --num_train_epochs 20 --logging_strategy epoch --save_strategy epoch --save_total_limit 10  \
# --load_best_model_at_end True --metric_for_best_model rouge2 --dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
# --generation_num_beams  16 \
# --eval_accumulation_steps 800 \
# --generation_num_return_sequences 16 \
# --generation_num_beam_groups 16 \
# --generation_diversity_penalty 1 \
# --model_name_or_path saves/cnndm/pytorch_model.bin \
# --config_name saves/cnndm/config.json \
# --tokenizer_name saves/cnndm/ \
# --max_source_length 1000 --max_target_length 100 --disable_tqdm True \
# --output_dir diverse-test

# python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
# --dataset_name cnndm \
# --split dev \
# --do_predict --evaluation_strategy epoch --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 \
# --learning_rate 5e-5 --num_train_epochs 20 --logging_strategy epoch --save_strategy epoch --save_total_limit 10  \
# --load_best_model_at_end True --metric_for_best_model rouge2 --dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
# --generation_num_beams  32 \
# --eval_accumulation_steps 800 \
# --generation_num_return_sequences 32 \
# --generation_num_beam_groups 32 \
# --generation_diversity_penalty 1 \
# --model_name_or_path saves/cnndm/pytorch_model.bin \
# --config_name saves/cnndm/config.json \
# --tokenizer_name saves/cnndm/ \
# --max_source_length 1000 --max_target_length 100 --disable_tqdm True \
# --output_dir diverse-test


# for i in {2..40} 
# do 
#     beam=1
#     if [ $i -lt 4 ]; then
#         beam=4
#     else 
#         beam=$i
#     fi 
#     python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
#     --dataset_name cnndm \
#     --split test \
#     --do_predict --evaluation_strategy epoch --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 \
#     --learning_rate 5e-5 --num_train_epochs 20 --logging_strategy epoch --save_strategy epoch --save_total_limit 10  \
#     --load_best_model_at_end True --metric_for_best_model rouge2 --dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
#     --generation_num_beams $beam  \
#     --generation_num_return_sequences $i \
#     --generation_num_beam_groups $beam \
#     --generation_diversity_penalty 1 \
#     --eval_accumulation_steps 800 \
#     --model_name_or_path saves/cnndm/pytorch_model.bin \
#     --config_name saves/cnndm/config.json \
#     --tokenizer_name saves/cnndm/ \
#     --max_source_length 1000 --max_target_length 100 --disable_tqdm True \
#     --output_dir diverse-test

# done