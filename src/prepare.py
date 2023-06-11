import re
import random
import mazelib as mzl
import array
import numpy as np

TOKENS = [
    "0",
    "1",
    "\n",
    "<start>",
    "<end>"
]

algorithms = mzl.algorithms
ALLOWED_ALGORITHMS = [
    algorithms.Prims,
    algorithms.AldousBroder,
    algorithms.BacktrackingGenerator,
    algorithms.BinaryTree,
]


def encode(s: str):
    """
    Creates a mapping of tokens to their int representations.
    Int representation is provided from the position in the TOKENS list.
    """
    for i, token in enumerate(TOKENS):
        s = s.replace(token, i.__str__())
    return list(map(int, list(s)))


def decode(tokens: list[TOKENS]):
    """
    Decodes the given list of int tokens into actual string representation.
    """
    decoded_tokens = [TOKENS[token] for token in tokens]
    return ''.join(decoded_tokens)

def create_maze():
    # Select a random algorithm
    algorithm_index = random.randint(0, len(ALLOWED_ALGORITHMS) - 1)
    selected_algorithm = ALLOWED_ALGORITHMS[algorithm_index]

    # Create the Maze
    maze = mzl.Maze()
    maze.generator = selected_algorithm(10, 10)
    maze.generate()

    #
    maze.string_to_maze(decode(encode(maze.__str__())))
    maze.display_maze()


if __name__ == '__main__':
    create_maze()
    print("Foobar")
