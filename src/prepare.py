from datasets import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from src.config import MazeAIConfig
from src.data import MazeAIData
from src.util import bordered, rooted


class MazeAIPrepare:
    """
    Builds dataset of mazes for training the model.
    Uses built data to then generate a tokenizer.
    Both dataset and tokenizer are saved to the output folder.
    """
    def __init__(self, config: MazeAIConfig):
        self.config = config

        # Training dataset
        self.build_dataset()
        self.training_data = self.load_dataset()

        # Create tokenizer and it's iterator for dataset
        self.tokenizer = ByteLevelBPETokenizer()
        self.batch_iterator = self.batch_iterator()

        # Train ans save tokenizer
        self.train_tokenizer()
        self.save_tokenizer()

    def build_dataset(self):
        """Generate the training binary data"""
        print(bordered("Generating Mazes"))
        data = MazeAIData(self.config)
        data.generate()
        print()

    def load_dataset(self):
        """Load training data for tokenizer"""
        print(bordered("Load Mazes For Training"))
        training_data = load_dataset(rooted(self.config.output.data))["train"]
        print()
        return training_data

    def batch_iterator(self):
        """Create iterator for moving through training data"""
        batch_size = self.config.tokenizer.batch_size
        batch_range = range(0, len(self.training_data), batch_size)
        return (self.training_data[i: i + batch_size]["text"] for i in batch_range)

    def train_tokenizer(self):
        """Train the tokenizer"""
        tokenizer_config = self.config.tokenizer
        print(bordered("Training Tokenizer"))
        self.tokenizer.train_from_iterator(
            self.batch_iterator,
            show_progress=True,
            min_frequency=tokenizer_config.min_frequency,
            vocab_size=tokenizer_config.vocab_size,
            special_tokens=[tokenizer_config.pad_token, tokenizer_config.mask_token]
        )

    def save_tokenizer(self):
        """Save tokenizer to a file"""
        print("Saving tokenizer at " + self.config.output.tokenizer)
        self.tokenizer.save(rooted(self.config.output.tokenizer))
