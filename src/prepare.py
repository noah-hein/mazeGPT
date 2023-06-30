from datasets import load_dataset

from config import MazeAIConfig
from data import MazeAIData
from tokenizer import MazeAITokenizer


def prepare():
    """
    Builds dataset of mazes for training the model.
    Uses built data to then generate a tokenizer.
    Both dataset and tokenizer are saved to the output folder.
    """
    # Choose the config
    config = MazeAIConfig()

    # Generate the training binary_tree
    data = MazeAIData(config)
    data.generate()

    # Build the tokenizer based on data
    training_data = load_dataset(config.DATA_DIRECTORY)["train"]
    tokenizer = MazeAITokenizer(config)
    tokenizer.train_from_data(training_data)
    tokenizer.save_to_file()


if __name__ == '__main__':
    """
    Allows the client to run the prepare script without the cli.
    """
    prepare()
