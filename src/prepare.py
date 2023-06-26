from datasets import load_dataset

from config import MazeAIConfig
from data import MazeAIData
from tokenizer import MazeAITokenizer

if __name__ == '__main__':
    # Choose the config
    config = MazeAIConfig()

    # Generate the training mazes
    data = MazeAIData(config)
    data.generate()

    # Build the tokenizer based on data
    training_data = load_dataset(config.DATA_DIRECTORY)["train"]
    tokenizer = MazeAITokenizer(config)
    tokenizer.train_from_data(training_data)
    tokenizer.save_to_file()
