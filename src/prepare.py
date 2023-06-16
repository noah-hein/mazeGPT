import os
import pathlib
import random

from tokenizers.implementations import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast

import mazelib as mzl
import mazelib as algorithms

TOKENS = [
    "0",
    "1",
    "<br>"
    "<start>",
    "<end>"
]

ALLOWED_ALGORITHMS = [
    algorithms.Prims,
    algorithms.AldousBroder,
    algorithms.BacktrackingGenerator,
    algorithms.BinaryTree,
]

NUMBER_OF_MAZES = 1000
TRAINING_PERCENT = 0.9

MIN_HEIGHT = 10
MAX_HEIGHT = 10
MIN_WIDTH = 10
MAX_WIDTH = 10

OUTPUT_DIRECTORY = "../data"
TRAIN_FILENAME = "train.bin"
VALIDATION_FILENAME = "validation.bin"


def generate_maze():
    # Maze parameters
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    seed = random.randint(0, 10000)

    # Select a random algorithm
    algorithm_index = random.randint(0, len(ALLOWED_ALGORITHMS) - 1)
    selected_algorithm = ALLOWED_ALGORITHMS[algorithm_index]

    # Create the Maze
    maze = mzl.Maze(seed)
    maze.generator = selected_algorithm(height, width)
    maze.generate()

    # Generate build log and return
    maze_description = "| {: >10} | {: >25} | {: >20} |"
    maze_description = maze_description.format(
        width.__str__() + "x" + height.__str__(),
        selected_algorithm.__name__,
        seed
    )
    print(maze_description)
    return maze


def generate_mazes():
    table_header = "| {: >10} | {: >25} | {: >20} |".format("dimensions", "algorithm", "seed")
    print(table_header)
    print(len(table_header) * "-")
    return [generate_maze() for _ in range(NUMBER_OF_MAZES)]


def get_tokenizer_data(tokenizer_data):
    for i in range(0, len(tokenizer_data), 5):
        yield ''.join(str(_) for _ in tokenizer_data[i: i + 5])


def build_tokenizer(tokenizer_data):
    # Use Uni-gram sentence piece model
    sp_tokenizer = SentencePieceBPETokenizer()
    sp_tokenizer.train_from_iterator(
        tokenizer_data,
        vocab_size=30000,
        min_frequency=5,
        show_progress=True,
        limit_alphabet=500,
        special_tokens=["<start>", "<end>"]
    )

    # Save model definition to file
    tokenizer_path = os.path.join(OUTPUT_DIRECTORY, "tokenizer.json")
    sp_tokenizer.save(tokenizer_path)
    return sp_tokenizer


if __name__ == '__main__':
    # Create training data
    mazes = generate_mazes()
    data = "".join(str(maze) for maze in mazes)
    data_length = len(data)

    # Split off some percent of training data for validation
    training_data = data[:int(data_length * TRAINING_PERCENT)]
    validation_data = data[int(data_length * TRAINING_PERCENT):]

    # Create tokenizer
    tokenizer = build_tokenizer(get_tokenizer_data(mazes))

    # Encode training and validation data
    enc = tokenizer.encode(mazes[0].__str__())
    train_ids = tokenizer.encode(training_data).ids
    validation_ids = tokenizer.encode(validation_data).ids

    # # Create output locations
    pathlib.Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    output_directory = os.path.join(os.path.dirname(__file__), OUTPUT_DIRECTORY)
    train_file_path = os.path.join(output_directory, TRAIN_FILENAME)
    validation_file_path = os.path.join(output_directory, VALIDATION_FILENAME)

    # Save training and validation data to files
    #train_ids.tofile(train_file_path)
    #validation_ids.tofile(validation_file_path)
