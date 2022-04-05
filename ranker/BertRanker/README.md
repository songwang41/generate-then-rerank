# Bertranker
这是一个single tower，独立训练的reranker模型

## Data preparation
运行 `python preprocess.py` 可获得训练reranker模型的数据。需要修改该py文件中的参数，例如 candidate_dir修改为模型生成的candidate文件的路径，num_cand为生成候选的数量。

google drive中上传了5000个数据的cnndm_debug，完整数据未上传。其中：
```markdown
├── cnndm_debug
   └── train         # 使用bart-base在cnndm整体训练集上训练，并在训练集前5000个数据上生成candidaite
        ├── gen_from_16 # 生成包含16个candidate的训练数据
   └── dev      # 使用bart-base在cnndm整体训练集上训练，并验证集集前5000个数据上生成candidaite
   └── test       # 使用bart-base在cnndm整体训练集上训练，并在测试集前5000个数据上生成candidaite
   └── train_half       # 使用bart-base在cnndm的后半部分训练集上(后143557个样本)训练，并在训练集前5000个数据上生成candidaite
   └── train_sample       # 使用bart-base在cnndm整体训练集上训练，并在训练集前5000个数据上生成candidaite，但是替换Decoding strategy为top-k sampling
   └── dev_sample       # 使用bart-base在cnndm整体训练集上训练，并在验证集前5000个数据上生成candidaite，但是替换Decoding strategy为top-k sampling
   └── test_sample       # 使用bart-base在cnndm整体训练集上训练，并在测试集前5000个数据上生成candidaite，但是替换Decoding strategy为top-k sampling

```

## Training
`bash run_train.sh`
