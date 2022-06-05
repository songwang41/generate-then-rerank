python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path /weizhou_data/models/unixcoder-base \
	--train_filename data/concode/train.json \
	--dev_filename data/concode/dev.json \
	--output_dir saved_models \
	--max_source_length 350 \
	--max_target_length 150 \
	--beam_size 3 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 30 


python run.py \
	--do_test \
	--model_name_or_path /weizhou_data/models/unixcoder-base \
	--test_filename data/concode/test.json \
	--output_dir saved_models \
	--max_source_length 350 \
	--max_target_length 150 \
	--beam_size 3 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 30 



python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path /weizhou_data/models/unixcoder-base \
	--train_filename data/concode/train.json \
	--dev_filename data/concode/dev.json \
	--output_dir save_gpu \
	--max_source_length 350 \
	--max_target_length 150 \
	--beam_size 3 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 1000 