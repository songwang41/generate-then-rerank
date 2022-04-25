from transformers import BartConfig, BartTokenizer
from model_utils.generation_utils import BartForConditionalGeneration


config = BartConfig.from_pretrained(
        '/weizhou_data/models/bart-base',
        cache_dir=None,
        revision='main',
        use_auth_token=False ,
    )
tokenizer = BartTokenizer.from_pretrained(
    '/weizhou_data/models/bart-base',
    cache_dir=None,
    use_fast=True,
    revision='main',
    use_auth_token=False ,
)
model = BartForConditionalGeneration.from_pretrained(
    '/weizhou_data/models/bart-base',
    from_tf=False,
    config=config,
    cache_dir=None,
    revision='main',
    use_auth_token=False ,
)


input_context = ["My cute dog", "Hello world"]
# get tokens of words that should not be generated
bad_words_ids = [tokenizer(bad_word, add_prefix_space=True).input_ids for bad_word in ["idiot", "stupid", "shut up"]]
# encode input context
input_ids = tokenizer(input_context, padding = True, return_tensors="pt").input_ids
print(input_ids)

# generate sequences without allowing bad_words to be generated
outputs = model.generate_ours(input_ids=input_ids, max_length=20, do_sample=True, bad_words_ids=bad_words_ids, num_beams=1)
# print(outputs.scores[0])
# print(outputs.scores[1])
print(outputs.seq_lens)

print("Generated:", tokenizer.decode(outputs.sequences[0], skip_special_tokens=False))
print("Generated:", tokenizer.decode(outputs.sequences[1], skip_special_tokens=False))