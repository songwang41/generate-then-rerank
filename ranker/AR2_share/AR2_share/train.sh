# a naive example of triviaqa training.  Model_After_ICT is the checkpoint trained with ict. 
# The training data can
cd /quantus-nfs/zh/AR2_share
EXP_NAME=run_de_ict_triviaqa_share_test
BASE_DIR=/mnt/data/denseIR
Model_After_ICT=./model_optim_rng.pt
BASE_SCRIPT_DIR=/quantus-nfs/zh/AR2_share

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9539 \
$BASE_SCRIPT_DIR/wiki/run_de_model_ernie.py \
--model_type="nghuyong/ernie-2.0-en" \
--origin_data_dir=$BASE_DIR/data/trivia_data/biencoder-trivia-train.json \
--origin_data_dir_dev=$BASE_DIR/data/trivia_data/biencoder-trivia-dev.json \
--model_name_or_path_ict=/mnt/data/Megatron-LM/ict_ernie_base/iter_0112000/mp_rank_00/model_optim_rng.pt \
--max_seq_length=256 --per_gpu_train_batch_size=16 --gradient_accumulation_steps=1 \
--learning_rate=2e-5 --output_dir $BASE_DIR/ckpt/$EXP_NAME \
--warmup_steps 4000 --logging_steps 100 --save_steps 1000 --max_steps 40000 \
--log_dir ../tensorboard/logs/run_de_model_ict_ernie_triviaqa_debug --fp16 \
--number_neg 1 


# Step 2 inference de_0
EXP_NAME=run_de_model_ict_ernie_triviaqa
BASE_DIR=/mnt/data/denseIR
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9539 \
$BASE_SCRIPT_DIR/wiki/inference_de_wiki_gpu.py \
--model_type="nghuyong/ernie-2.0-en" \
--eval_model_dir $BASE_DIR/ckpt/$EXP_NAME/checkpoint-40000 \
--output_result_list_dir=$BASE_DIR/ckpt/$EXP_NAME/1k/triviaqa-test_ernie.pkl \
--output_dir $BASE_DIR/ckpt/$EXP_NAME/1k \
--test_qa_path $BASE_DIR/data/trivia_data/trivia-test.qa.csv \
--train_qa_path $BASE_DIR/data/trivia_data/trivia-train.qa.csv \
--dev_qa_path $BASE_DIR/data/trivia_data/trivia-dev.qa.csv \
--max_seq_length=256 --per_gpu_eval_batch_size 1024 \
--passage_idx_cache_path=$BASE_DIR/data/QA_NQ_data \
--passage_path=$BASE_DIR/data/psgs_w100.tsv --fp16

# Step 3 generate reranker training data
python $BASE_SCRIPT_DIR/utils/prepare_ce_data.py $BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/dev_result_dict_list.json \
$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/dev_ce_0_triviaqa.json \
$BASE_DIR/data/trivia_data/biencoder-trivia-dev.json

python $BASE_SCRIPT_DIR/utils/prepare_ce_data.py \
$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/train_result_dict_list.json \
$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/train_ce_0_triviaqa.json \
$BASE_DIR/data/trivia_data/biencoder-trivia-train.json


# Step 4 warming reranker
mkdir $BASE_DIR/ckpt/run_ce_model_after_ict_ernielarge_triviaqa_2
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9538 \
$BASE_SCRIPT_DIR/wiki/run_ce_model_ernie.py \
--model_type=nghuyong/ernie-2.0-large-en --max_seq_length=256 \
--per_gpu_train_batch_size=1 --gradient_accumulation_steps=8 \
--number_neg=15 --learning_rate=1e-5 \
--output_dir=$BASE_DIR/ckpt/run_ce_model_after_ict_ernielarge_triviaqa_2 \
--origin_data_dir=$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/train_ce_0_triviaqa.json \
--origin_data_dir_dev=$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/dev_ce_0_triviaqa.json \
--warmup_steps=1000 --logging_steps=100 --save_steps=1000 \
--max_steps=10000 --log_dir=tensorboard/logs/run_ce_model_after_ict_ernielarge_triviaqa_2 \
--fp16



EXP_NAME=co_training_triviaqa_share_test
Warm_Reranker_PATH=$BASE_DIR/ckpt/run_ce_model_after_ict_ernielarge_triviaqa/checkpoint-4000
Warm_de_path=$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/checkpoint-40000
Reranker_TYPE=nghuyong/ernie-2.0-large-en
Iteration_step=2000
Iteration_reranker_step=500
MAX_STEPS=32000
# for global_step in `seq 0 2000 $MAX_STEPS`; do echo $global_step; done;
for global_step in `seq 0 $Iteration_step $MAX_STEPS`; 
do 
    python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9539 $BASE_SCRIPT_DIR/wiki/co_training_wiki_train.py \
    --model_type=nghuyong/ernie-2.0-en \
    --model_name_or_path=$Warm_de_path \
    --max_seq_length=128 --per_gpu_train_batch_size=8 --gradient_accumulation_steps=1 \
    --number_neg=15 --learning_rate=1e-5 \
    --reranker_model_type=$Reranker_TYPE \
    --reranker_model_path=$Warm_Reranker_PATH \
    --reranker_learning_rate=1e-6 \
    --output_dir=$BASE_DIR/ckpt/$EXP_NAME \
    --log_dir=tensorboard/logs/$EXP_NAME \
    --origin_data_dir=$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/train_ce_0_triviaqa.json \
    --origin_data_dir_dev=$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/dev_ce_0_triviaqa.json \
    --warmup_steps=2000 --logging_steps=10 --save_steps=2000 --max_steps=$MAX_STEPS \
    --gradient_checkpointing --normal_loss \
    --iteration_step=$Iteration_step \
    --iteration_reranker_step=$Iteration_reranker_step \
    --temperature_normal=1 --ann_dir=$BASE_DIR/ckpt/$EXP_NAME/temp --adv_lambda 0.5 --global_step=$global_step

    g_global_step=`expr $global_step + $Iteration_step`
    python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9539 $BASE_SCRIPT_DIR/wiki/co_training_wiki_generate.py \
    --model_type=nghuyong/ernie-2.0-en \
    --model_name_or_path=$Warm_de_path \
    --max_seq_length=128 --per_gpu_train_batch_size=8 --gradient_accumulation_steps=1 \
    --number_neg=15 --learning_rate=1e-5 \
    --reranker_model_type=$Reranker_TYPE \
    --reranker_model_path=$Warm_Reranker_PATH \
    --reranker_learning_rate=1e-6 \
    --output_dir=$BASE_DIR/ckpt/$EXP_NAME \
    --log_dir=tensorboard/logs/$EXP_NAME \
    --origin_data_dir=$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/train_ce_0_triviaqa.json \
    --origin_data_dir_dev=$BASE_DIR/ckpt/run_de_model_ict_ernie_triviaqa/40k/dev_ce_0_triviaqa.json \
    --train_qa_path=$BASE_DIR/data/trivia_data/trivia-train.qa.csv \
    --test_qa_path=$BASE_DIR/data/trivia_data/trivia-test.qa.csv \
    --dev_qa_path=$BASE_DIR/data/trivia_data/trivia-dev.qa.csv \
    --passage_path=$BASE_DIR/data/psgs_w100.tsv \
    --warmup_steps=2000 --logging_steps=10 --save_steps=2000 --max_steps=$MAX_STEPS \
    --gradient_checkpointing --normal_loss --adv_step=0 \
    --iteration_step=$Iteration_step \
    --iteration_reranker_step=$Iteration_reranker_step \
    --temperature_normal=1 --ann_dir=$BASE_DIR/ckpt/$EXP_NAME/temp --adv_lambda=0.5 --global_step=$g_global_step
done
