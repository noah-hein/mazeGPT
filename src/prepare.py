import os
import pathlib
import random
import mazelib as mzl
import mazelib as algorithms

from src.mazelib import Maze
from tokenizers.implementations import SentencePieceBPETokenizer

SPECIAL_TOKENS = [
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
TRAIN_FILENAME = "train.txt"
VALIDATION_FILENAME = "validation.txt"
TOKENIZER_FILENAME = "tokenizer.json"


def generate_maze(index: int):
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
    maze_description = "| {: >10} | {: >10} | {: >25} | {: >20} |"
    maze_description = maze_description.format(
        index.__str__(),
        width.__str__() + "x" + height.__str__(),
        selected_algorithm.__name__,
        seed
    )
    print(maze_description)
    return maze


def generate_mazes():
    print("Generating mazes")
    table_header = "| {: >10} | {: >10} | {: >25} | {: >20} |".format("i", "dimensions", "algorithm", "seed")
    print(table_header)
    print(len(table_header) * "-")
    mazes = [generate_maze(i) for i in range(NUMBER_OF_MAZES)]
    return mazes


def build_dataset(mazes: list[Maze]):
    # Combine all mazes into a string
    data = "".join(str(maze) for maze in mazes)
    data_length = len(data)

    # Split off some percent of training data for validation
    print("Splitting training and validation datasets")
    training_data = data[:int(data_length * TRAINING_PERCENT)]
    validation_data = data[int(data_length * TRAINING_PERCENT):]

    # Create output locations
    pathlib.Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    output_directory = os.path.join(os.path.dirname(__file__), OUTPUT_DIRECTORY)
    train_file_path = os.path.join(output_directory, TRAIN_FILENAME)
    validation_file_path = os.path.join(output_directory, VALIDATION_FILENAME)

    # Save training and validation data to files
    print("Saving training data to " + TRAIN_FILENAME)
    print("saving validation data to " + VALIDATION_FILENAME)
    print(training_data, file=open(train_file_path, "w"))
    print(validation_data, file=open(validation_file_path, "w"))


def get_tokenizer_data(tokenizer_data):
    for i in range(0, len(tokenizer_data), 5):
        yield ''.join(str(_) for _ in tokenizer_data[i: i + 5])


def build_tokenizer(mazes: list[Maze]):
    # Use Uni-gram sentence piece model
    print("Creating the tokenizer...")
    sp_tokenizer = SentencePieceBPETokenizer()
    sp_tokenizer.train_from_iterator(
        get_tokenizer_data(mazes),
        vocab_size=30000,
        min_frequency=5,
        show_progress=False,
        limit_alphabet=500,
        special_tokens=SPECIAL_TOKENS
    )

    # Save model definition to file
    print("Saving tokenizer at " + TOKENIZER_FILENAME)
    tokenizer_path = os.path.join(OUTPUT_DIRECTORY, TOKENIZER_FILENAME)
    sp_tokenizer.save(tokenizer_path)
    return sp_tokenizer


if __name__ == '__main__':
    generated_mazes = generate_mazes()
    build_dataset(generated_mazes)
    build_tokenizer(generated_mazes)

    # Encode training and validation data
    # training_ids = tokenizer.encode(training_data)
    # validation_ids = tokenizer.encode(validation_data)

    # Save training and validation data to files
    #train_ids.tofile(train_file_path)
    #validation_ids.tofile(validation_file_path)
