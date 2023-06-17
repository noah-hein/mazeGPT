import os
import pathlib
import random
import mazelib as mzl
import src.config as config

from src.mazelib import Maze
from tokenizers.implementations import SentencePieceBPETokenizer


def generate_maze(index: int):
    # Maze parameters
    height = random.randint(config.MIN_HEIGHT, config.MAX_HEIGHT)
    width = random.randint(config.MIN_WIDTH, config.MAX_WIDTH)
    seed = random.randint(0, 10000)

    # Select a random algorithm
    algorithm_index = random.randint(0, len(config.ALLOWED_ALGORITHMS) - 1)
    selected_algorithm = config.ALLOWED_ALGORITHMS[algorithm_index]

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
    mazes = [generate_maze(i) for i in range(config.NUMBER_OF_MAZES)]
    return mazes


def build_dataset(mazes: list[Maze]):
    # Combine all mazes into a string
    data = "".join(str(maze) for maze in mazes)

    # Create output for file
    pathlib.Path(config.OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    output_directory = os.path.join(os.path.dirname(__file__), config.OUTPUT_DIRECTORY)
    data_file_path = os.path.join(output_directory, config.DATA_FILENAME)

    # Save training and validation data to files
    print("Saving dataset data to " + config.DATA_FILENAME)
    print(data, file=open(data_file_path, "w"))


def get_tokenizer_data(tokenizer_data):
    for i in range(0, len(tokenizer_data), 5):
        yield ''.join(str(_) for _ in tokenizer_data[i: i + 5])


def build_tokenizer(mazes: list[Maze]):
    # Use Uni-gram sentence piece model
    print("Creating the tokenizer...")
    sp_tokenizer = SentencePieceBPETokenizer()
    sp_tokenizer.train_from_iterator(
        get_tokenizer_data(mazes),
        vocab_size=100,
        min_frequency=5,
        show_progress=False,
        limit_alphabet=500,
        special_tokens=config.SPECIAL_TOKENS
    )

    # Save model definition to file
    print("Saving tokenizer at " + config.TOKENIZER_FILENAME)
    tokenizer_path = os.path.join(config.OUTPUT_DIRECTORY, config.TOKENIZER_FILENAME)
    sp_tokenizer.save(tokenizer_path)


if __name__ == '__main__':
    generated_mazes = generate_mazes()
    build_dataset(generated_mazes)
    build_tokenizer(generated_mazes)
