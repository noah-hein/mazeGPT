from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from config import MazeAIConfig
from tokenizers.implementations import ByteLevelBPETokenizer


class MazeAiTokenizer(ByteLevelBPETokenizer):

    def __init__(self, config: MazeAIConfig):
        self.config = config
        self.pad_token = config.PAD_TOKEN
        self.mask_token = config.MASK_TOKEN
        super().__init__()

    def load(self):
        return PreTrainedTokenizerFast(tokenizer_file=self.config.TOKENIZER_FILE_PATH)

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
