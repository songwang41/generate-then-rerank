from datasets import  load_metric
from nltk import word_tokenize
from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained('/weizhou_data/models/bart-base')
bleu = load_metric('bleu')


# predictions = [
#  ["hi", "there", "universal", "kenobi"],                             # tokenized prediction of the first sample
#  ["foo", "bar", "foobar"]                                             # tokenized prediction of the second sample
#  ]

# references = [
# [["hello", "there", "general", "kenobi"], ["hello", "there", "!"]],  # tokenized references for the first sample (2 references)
# [["foo", "bar", "foobar"]]                                           # tokenized references for the second sample (1 reference)
# ]

# for i in range(1, 5):
#     print(bleu._compute(predictions, references, max_order = i))

sequences = [
    'Best played bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard" He died Monday after a brief illness.',
    'Donna Brazile: Baltimore rioting is a reminder of the divide between the left and right.'
]

input_tokens = [tokenizer.tokenize(s) for s in sequences]
print(input_tokens)

input_ids = [tokenizer.convert_tokens_to_ids(s) for s in input_tokens]

print(tokenizer.batch_decode(input_ids, clean_up_tokenization_spaces=True))

print(tokenizer.batch_decode(input_ids, clean_up_tokenization_spaces=False))

print([word_tokenize(s) for s in sequences])