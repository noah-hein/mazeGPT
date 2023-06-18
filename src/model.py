import os
from src import config
from transformers import \
    PreTrainedTokenizerFast, \
    AutoConfig, GPT2LMHeadModel

# Load in the pretrained tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.TOKENIZER_FILE_PATH)
tokenizer.pad_token = "[PAD]"
tokenizer.mask_token = "[MASK]"

# Define the model
model_config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(model_config)
