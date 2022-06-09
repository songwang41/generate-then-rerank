# for dev and test, use beam search
python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
--dataset_name xsum \
--split dev \
--use_tokenized_data False \
--do_predict --per_device_eval_batch_size 8 \
--dataloader_pin_memory True --predict_with_generate True --generation_max_length 109 \
--eval_accumulation_steps 800 \
--generation_do_sample False \
--generation_num_beams 16 \
--generation_num_return_sequences 16 \
--generation_num_beam_groups 16 \
--generation_diversity_penalty 1.0 \
--model_name_or_path saves/bart-large-xsum \
--config_name saves/bart-large-xsum \
--tokenizer_name saves/bart-large-xsum \
--max_source_length 1020 --max_target_length 109 --disable_tqdm False \
--output_dir diverse-test --save_name xsum-large-diverse

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
--dataset_name xsum \
--split test \
--use_tokenized_data False \
--do_predict --per_device_eval_batch_size 8 \
--dataloader_pin_memory True --predict_with_generate True --generation_max_length 109 \
--eval_accumulation_steps 800 \
--generation_do_sample False \
--generation_num_beams 16 \
--generation_num_return_sequences 16 \
--generation_num_beam_groups 16 \
--generation_diversity_penalty 1.0 \
--model_name_or_path saves/bart-large-xsum \
--config_name saves/bart-large-xsum \
--tokenizer_name saves/bart-large-xsum \
--max_source_length 1020 --max_target_length 109 --disable_tqdm False \
--output_dir diverse-test --save_name xsum-large-diverse