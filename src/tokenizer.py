import config

from maze import Maze
from tokenizers.implementations import ByteLevelBPETokenizer

class MazeTokenizer:


    def get_tokenizer_data(self, tokenizer_data):
        for i in range(0, len(tokenizer_data), 5):
            yield ''.join(str(_) for _ in tokenizer_data[i: i + 5])

    def build(self, mazes: list[Maze]):
        # Use Uni-gram sentence piece model
        print("Creating the tokenizer...")
        sp_tokenizer = ByteLevelBPETokenizer()
        sp_tokenizer.train_from_iterator(
            self.get_tokenizer_data(mazes),
            vocab_size=100,
            min_frequency=5,
            show_progress=False,
            special_tokens=config.SPECIAL_TOKENS
        )

        # Save model definition to file
        print("Saving tokenizer at " + config.TOKENIZER_FILENAME)
        sp_tokenizer.save(config.TOKENIZER_FILE_PATH)