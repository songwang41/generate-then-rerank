# test runing
python main.py --cuda --gpuid 0 1 2 3 4 5 6 7 -l \
  --log_name samsum-from32 \
  --model_name_or_path /weizhou_data/models/roberta-base \
  --cache_data \
  --train_batch_size 4 \
  --accumulate_step 2 \
  --report_freq 500 \
  --eval_freq 500 \
  --dataset samsum \
  --max_num 32 \
  --num_beams 32 \
  --epoch 10 \
  --warmup_steps 2000 