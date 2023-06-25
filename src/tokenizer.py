from datasets import load_dataset
from config import MazeAiConfig
from tokenizers.implementations import ByteLevelBPETokenizer


class MazeAiTokenizer:

    def __init__(self, config: MazeAiConfig):
        self.config = config

    def train(self):
        # Load maze data
        print("Training tokenizer from " + self.config.DATA_DIRECTORY)
        dataset = load_dataset(self.config.DATA_DIRECTORY)
        training_data = dataset["train"]

        # Create new tokenizer
        print("Creating the tokenizer...")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            training_data,
            vocab_size=0,
            show_progress=False,
            special_tokens=self.config.SPECIAL_TOKENS
        )

        # Save tokenizer model definition to file
        print("Saving tokenizer at " + self.config.TOKENIZER_FILENAME)
        tokenizer.save(self.config.TOKENIZER_FILE_PATH)
