from config import MazeAIConfig
from transformers import \
    PreTrainedTokenizerFast, \
    GPT2LMHeadModel, GPT2Config


class MazeAIModel(GPT2LMHeadModel):
    # ==================================================================================================================
    #       Constructor
    # ==================================================================================================================

    def __init__(self, config: MazeAIConfig):
        self.config = config
        self.tokenizer = self._load_tokenizer()
        self.model_config = self._build_model_config()
        super().__init__(self.model_config)

    # ==================================================================================================================
    #       Private Methods
    # ==================================================================================================================

    def _build_model_config(self):
        tokenizer = self.tokenizer
        return GPT2Config.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    def _load_tokenizer(self):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.config.TOKENIZER_FILE_PATH)
        tokenizer.pad_token = self.config.PAD_TOKEN
        tokenizer.mask_token = self.config.MASK_TOKEN
        return tokenizer
