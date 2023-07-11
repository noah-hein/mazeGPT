from datasets import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from src.config.default import MazeAIConfig
from src.data import MazeAIData


def prepare(config: MazeAIConfig):
    """
    Builds dataset of mazes for training the model.
    Uses built data to then generate a tokenizer.
    Both dataset and tokenizer are saved to the output folder.
    """
    # Generate the training binary_tree
    data = MazeAIData(config)
    data.generate()

    # Get training data for tokenizer
    training_data = load_dataset(config.data_directory())["train"]
    tokenizer = ByteLevelBPETokenizer()

    # Create iterator for moving through training data
    batch_range = range(0, len(training_data), config.BATCH_SIZE)
    batch_iterator = (training_data[i: i + config.BATCH_SIZE]["text"] for i in batch_range)

    # Build special tokens list
    dimension_tokens = data.dimension_tokens()
    special_tokens = config.SPECIAL_TOKENS + dimension_tokens

    # Train the tokenizer
    tokenizer.train_from_iterator(
        batch_iterator,
        show_progress=True,
        min_frequency=config.TOKENIZER_MIN_FREQUENCY,
        vocab_size=config.VOCAB_SIZE,
        special_tokens=special_tokens
    )

    # Save tokenizer to a file
    print("Saving tokenizer at " + config.TOKENIZER_FILENAME)
    tokenizer.save(config.tokenizer_path())


if __name__ == '__main__':
    """Allows the you to run the 'prepare' script without the CLI"""
    prepare(MazeAIConfig())
