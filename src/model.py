import os
from src import config
from transformers import \
    PreTrainedTokenizerFast, \
    AutoConfig, GPT2LMHeadModel

# Find the tokenizer
tokenizer_json_path = os.path.join(config.OUTPUT_DIRECTORY, config.TOKENIZER_FILENAME)

# Load in the pretrained tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json_path)
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
