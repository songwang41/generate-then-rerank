
from transformers import AutoTokenizer, AutoModel
import numpy as np
import fire


# from transformers import RobertaTokenizer, RobertaModel
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# model = RobertaModel.from_pretrained('roberta-base')
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)


def create_long_model(save_model_to, model_name_or_path="roberta-base", max_pos=1024):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)    
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape #514
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    config._name_or_path= config._name_or_path + "-"+ str(max_pos)
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos-2 # 512
    print("k", k, "step", step, "weight.shape", model.embeddings.position_embeddings.weight.shape)
    while k < max_pos - 1:
        print("k", k, new_pos_embed.shape)
        new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight[2:]
        k += step
    model.embeddings.position_embeddings.weight.data = new_pos_embed
    model.embeddings.position_ids = torch.from_numpy(np.arange(new_pos_embed.shape[0], dtype=np.int32)[np.newaxis, :])
    model.config = config
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    print("Saved model to", save_model_to)
    return None

if __name__=="__main__":
    fire.Fire(create_long_model)