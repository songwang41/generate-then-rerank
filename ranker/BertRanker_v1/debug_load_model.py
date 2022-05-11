from model_utils import RobertaRanker, BartRanker
from transformers import RobertaConfig, BartConfig


config = BartConfig.from_pretrained(
            'C:/Users/v-wshen/Desktop/Mypackages/projects/models/bart-base',
            revision='main',
            use_auth_token=True
        )

setattr(config, "loss_type", 'constractive')
setattr(config, "model_type", 'bart')

model = BartRanker.from_pretrained('C:/Users/v-wshen/Desktop/Mypackages/projects/models/bart-base', config = config)