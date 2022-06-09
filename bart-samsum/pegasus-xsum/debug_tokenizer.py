from lib2to3.pgen2 import token
from transformers import (
    PegasusConfig,
    PegasusTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
)

tokenizer = PegasusTokenizer.from_pretrained('/weizhou_data/models/pegasus-large')

s = "<s> I love china, i am from fujian province. "

print(tokenizer.tokenize(s))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s)))
print(tokenizer(s))