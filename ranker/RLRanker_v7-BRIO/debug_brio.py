from ipaddress import v4_int_to_packed
from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer
import torch
from brio_utils.model import BRIO

config = BartConfig.from_pretrained('C:/Users/v-wshen/Desktop/Mypackages/projects/models/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('C:/Users/v-wshen/Desktop/Mypackages/projects/models/bart-large-cnn')
model = BartForConditionalGeneration(config=config)

print(model.model.encoder.layers[2].fc2.bias)

model_t = BRIO('C:/Users/v-wshen/Desktop/Mypackages/projects/models/bart-large-cnn', tokenizer.pad_token_id, is_pegasus=False)
# for v in model_t.named_parameters():
#     # print(v)
#     # print(v[1].size())
#     if (50265 in v[1].size()):
#         print(v)
#         # print('+++++++++++++++++++++++++++++++++++\n')

print(model_t.model.model.encoder.embed_tokens.weight.size())
# model_t.model.resize_token_embeddings(len(tokenizer)) 
model_t.load_state_dict(torch.load('saves/BRIO/model_generation.bin', map_location=torch.device('cpu')))
model.load_state_dict(model_t.model.state_dict())

model.save_pretrained('saves')
print(model.model.encoder.layers[2].fc2.bias)