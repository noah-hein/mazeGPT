import os
import pathlib
import random
import mazelib as mzl
import mazelib.algorithms as algorithms
import numpy as np

TOKENS = [
    "0",
    "1",
    "\n",
    "<start>",
    "<end>"
]

ALLOWED_ALGORITHMS = [
    algorithms.Prims,
    algorithms.AldousBroder,
    algorithms.BacktrackingGenerator,
    algorithms.BinaryTree,
]

NUMBER_OF_MAZES = 10
TRAINING_PERCENT = 0.9

OUTPUT_DIRECTORY = "train"
TRAIN_FILENAME = "train.bin"
VALIDATION_FILENAME = "validation.bin"


def encode(s: str):
    """
    Creates a mapping of tokens to their int representations.
    Int representation is provided from the position in the TOKENS list.
    """
    for i, token in enumerate(TOKENS):
        s = s.replace(token, i.__str__())
    return np.array(list(map(int, list(s))))


def decode(tokens: list[TOKENS]):
    """
    Decodes the given list of int tokens into actual string representation.
    """
    decoded_tokens = [TOKENS[token] for token in tokens]
    return ''.join(decoded_tokens)

def create_maze_string():
    # Maze parameters
    height = 10
    width = 10
    seed = random.randint(0, 10000)

    # Select a random algorithm
    algorithm_index = random.randint(0, len(ALLOWED_ALGORITHMS) - 1)
    selected_algorithm = ALLOWED_ALGORITHMS[algorithm_index]

    # Create the Maze
    maze = mzl.Maze(seed)
    maze.generator = selected_algorithm(height, width)
    maze.generate()
    return maze.__str__()

def create_mazes_string():
    return "".join(create_maze_string() for _ in range(NUMBER_OF_MAZES))

if __name__ == '__main__':
    # Create training data
    data = create_mazes_string()
    data_length = len(data)

    # Split off some percent of training data for validation
    training_data = data[:int(data_length*TRAINING_PERCENT)]
    validation_data = data[int(data_length*TRAINING_PERCENT):]

    # Encode training and validation data
    train_ids = encode(training_data)
    validation_ids = encode(validation_data)

    # Create output locations
    pathlib.Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    output_directory = os.path.join(os.path.dirname(__file__), OUTPUT_DIRECTORY)
    train_file_path = os.path.join(output_directory, TRAIN_FILENAME)
    validation_file_path = os.path.join(output_directory, VALIDATION_FILENAME)

    # Save training and validation data to files
    train_ids.tofile(train_file_path)
    validation_ids.tofile(validation_file_path)

