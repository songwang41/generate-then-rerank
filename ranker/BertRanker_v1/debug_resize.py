from transformers import (
    RobertaConfig,
    RobertaTokenizer,
)

from model_utils import RobertaRanker

model_name_of_path = "/weizhou_data/models/roberta-base"
cache_dir = None
use_fast_tokenizer = True
model_revision = "main"
use_auth_token = False
loss_type = 'binary'

config = RobertaConfig.from_pretrained(
        model_name_of_path,
        cache_dir=cache_dir,
        revision=model_revision,
        use_auth_token=True if use_auth_token else None,
    )

setattr(config, "loss_type", loss_type)

tokenizer = RobertaTokenizer.from_pretrained(
    model_name_of_path,
    cache_dir=cache_dir,
    use_fast=use_fast_tokenizer,
    revision=model_revision,
    use_auth_token=True if use_auth_token else None,
)

model = RobertaRanker.from_pretrained(
    model_name_of_path,
    from_tf=bool(".ckpt" in model_name_of_path),
    config=config,
    cache_dir=cache_dir,
    revision=model_revision,
    use_auth_token=True if use_auth_token else None,
)

model._resize_position_embedding(1024)
print(model.roberta.embeddings.position_embeddings.weight.size())

print(model.roberta.embeddings.token_type_ids)
