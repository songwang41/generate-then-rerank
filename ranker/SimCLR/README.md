## preparation
因为在ranker训练的时候需要用到genrator的生成结果，所以需要把bart-base-diverse里面的results对应数据集的生成结果搬过来（这个结果由run_gen.sh生成）

## train
`bash run_trian.sh`
训练好的模型会保存在cache/model_name目录下


## evaluate 
`bash run_eval.sh` 
evaluate 一个训练好的模型，汇报在测试集上的rouge分数，同时也会生成ranking的top-1结果，保存在result目录下