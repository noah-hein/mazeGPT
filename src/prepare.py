import pathlib
import random
import config

from maze.algorithms import *
from maze import Maze, MazeFactory
from tokenizers.implementations import ByteLevelBPETokenizer


def build_dataset(mazes: list[Maze]):
    # Combine all mazes into a string
    data = "".join(str(m) for m in mazes)

    # Create output for file
    pathlib.Path(config.DATA_DIRECTORY).mkdir(parents=True, exist_ok=True)

    # Save training and validation data to files
    print("Saving dataset data to " + config.DATA_FILENAME)
    print(data, file=open(config.DATA_FILE_PATH, "w"))


def get_tokenizer_data(tokenizer_data):
    for i in range(0, len(tokenizer_data), 5):
        yield ''.join(str(_) for _ in tokenizer_data[i: i + 5])


def build_tokenizer(mazes: list[Maze]):
    # Use Uni-gram sentence piece model
    print("Creating the tokenizer...")
    sp_tokenizer = ByteLevelBPETokenizer()
    sp_tokenizer.train_from_iterator(
        get_tokenizer_data(mazes),
        vocab_size=100,
        min_frequency=5,
        show_progress=False,
        special_tokens=config.SPECIAL_TOKENS
    )

    # Save model definition to file
    print("Saving tokenizer at " + config.TOKENIZER_FILENAME)
    sp_tokenizer.save(config.TOKENIZER_FILE_PATH)


if __name__ == '__main__':
    # Generate the mazes
    mazeFactory = MazeFactory(config.MAX_HEIGHT, config.MAX_WIDTH, Prims)
    mazes = mazeFactory.generate(5)

    print(mazes[0])

    # generated_mazes = generate_mazes()
    # build_dataset(generated_mazes)
    # build_tokenizer(generated_mazes)
