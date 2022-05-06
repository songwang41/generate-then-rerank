
PRE_GEN=./saves/xsum/bart_diverse_20220428_learningrate_5e-5_nepochs_6_modeltype_large_dataset_xsum/checkpoint-19122/
SAVE_NAME=./saves/xsum/bart_diverse_20220428_learningrate_5e-5_nepochs_6_modeltype_large_dataset_xsum/candidates/bart-large-xsum
# for training set use sampling
python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
--dataset_name xsum \
--split train \
--do_predict --per_device_eval_batch_size 8 \
--dataloader_pin_memory True --predict_with_generate True --generation_max_length 109 \
--eval_accumulation_steps 800 \
--generation_do_sample True \
--generation_num_beams 1 \
--generation_num_return_sequences 16 \
--model_name_or_path $PRE_GEN \
--config_name $PRE_GEN \
--tokenizer_name $PRE_GEN \
--max_source_length 1020 --max_target_length 109 --disable_tqdm False \
--output_dir diverse-test 

# for dev and test, use beam search
python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
--dataset_name xsum \
--split dev \
--do_predict --per_device_eval_batch_size 8 \
--dataloader_pin_memory True --predict_with_generate True --generation_max_length 109 \
--eval_accumulation_steps 800 \
--generation_do_sample False \
--generation_num_beams 16 \
--generation_num_return_sequences 16 \
--model_name_or_path $PRE_GEN \
--config_name $PRE_GEN \
--tokenizer_name $PRE_GEN \
--max_source_length 1020 --max_target_length 109 --disable_tqdm False \
--output_dir diverse-test 


python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
--dataset_name xsum \
--split test \
--do_predict --per_device_eval_batch_size 8 \
--dataloader_pin_memory True --predict_with_generate True --generation_max_length 109 \
--eval_accumulation_steps 800 \
--generation_do_sample False \
--generation_num_beams 16 \
--generation_num_return_sequences 16 \
--model_name_or_path $PRE_GEN \
--config_name $PRE_GEN \
--tokenizer_name $PRE_GEN \
--max_source_length 1020 --max_target_length 109 --disable_tqdm False \
--output_dir diverse-test 

#--use_tokenized_data False \
#--save_name $SAVE_NAME