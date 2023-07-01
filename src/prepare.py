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

    # Build the tokenizer based on data
    training_data = load_dataset(config.DATA_DIRECTORY)["train"]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        training_data,
        vocab_size=0,
        show_progress=False,
        special_tokens=config.SPECIAL_TOKENS
    )

    # Save tokenizer to a file
    print("Saving tokenizer at " + config.TOKENIZER_FILENAME)
    tokenizer.save(config.TOKENIZER_FILE_PATH)


if __name__ == '__main__':
    """Allows the you to run the 'prepare' script without the CLI"""
    prepare(MazeAIConfig())
