# generate-then-rerank

## requirements
* cuda 11.0
* pytorch 1.10.1
* transformers 4.8.1
* datasets 1.12.1
* nltk
* rouge_score

or access the virtual environment with docker image <https://hub.docker.com/repository/docker/shenwzh/transformers>

## project overview

```markdown
├── genarate-then-rerank
   └── bart-mt          # 机器翻译，待开发
   └── bart-samsum      # bart摘要，现在samsum和cnn/dailymail都能跑
       ├── bart-base-diverse        # 基于huggingface transformers框架开发，修改了框架的代码，现在可以利用beam search生成多个结果(r un_diverse_gen.py)，但是训练的时候还是用beam search top-1的结果来做model selection
       ├── data        # 数据文件，有samsum 和cnn/dailymail

   └── ranker       # 训练ranker，目前还没有实现GAN的方法，都是generator和ranker分别训练
       ├── data        # 数据文件，现在暂时只有samsum
       ├── SimCLR        # 基于SimCLS <https://arxiv.org/abs/2106.01890>实现的ranker，方法基本和原文一致，使用generator生成的beam里面的文本作为candidate，计算sourceh和每个candidate的roberta编码表征的预先相似度，训练目标是ranking loss，ranking的groud truth是每个candidate的rouge分数排序，目前效果并不是非常理想
       ├── SimCLR        # SimCLS的改进，在candidate的选取方面，只随机保留generator生成的一半的文本作为hard negative，另外一半的candidate通过在其他的训练样例的生成结果中随机抽取，作为easy negative，让模型训练更容易，目前还没跑训练，不知道结果如何。
              
```