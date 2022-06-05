from transformers import RobertaTokenizer
from transformers import BartTokenizer
import nltk

# tokenizer = RobertaTokenizer.from_pretrained('C:/Users/v-wshen/Desktop/Mypackages/projects/models/roberta-base')
tokenizer = BartTokenizer.from_pretrained('C:/Users/v-wshen/Desktop/Mypackages/projects/models/bart-base')

sent = "I love China."
print(sent)
print(tokenizer.tokenize(sent))
print(nltk.word_tokenize(sent))
print(sent.split())