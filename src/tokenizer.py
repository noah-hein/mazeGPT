from transformers import PreTrainedTokenizerFast
from tokenizers.implementations import ByteLevelBPETokenizer
from src.config.base import MazeAIConfig


class MazeAITokenizer(ByteLevelBPETokenizer):

    def __init__(self, config: MazeAIConfig):
        self.config = config
        self.pad_token = config.PAD_TOKEN
        self.mask_token = config.MASK_TOKEN
        super().__init__()

    def load(self):
        loaded_tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.config.TOKENIZER_FILE_PATH)
        loaded_tokenizer.pad_token = self.pad_token
        loaded_tokenizer.mask_token = self.mask_token
        return loaded_tokenizer

    def train_from_data(self, training_data):
        print("Training the tokenizer")
        self.train_from_iterator(
            training_data,
            vocab_size=0,
            show_progress=False,
            special_tokens=self.config.SPECIAL_TOKENS
        )

    def save_to_file(self):
        print("Saving tokenizer at " + self.config.TOKENIZER_FILENAME)
        self.save(self.config.TOKENIZER_FILE_PATH)
