import random
import mazelib as mzl
import array
import numpy as np

algorithms = mzl.algorithms
allowed_algorithms = [
    algorithms.Prims,
    algorithms.AldousBroder,
    algorithms.BacktrackingGenerator,
    algorithms.BinaryTree,
]

tokens = [
    "<start>",
    "<end>",
    "1",
    "0",
    "\n"
]


def create_maze():
    # Select a random algorithm
    algorithm_index = random.randint(0, len(allowed_algorithms) - 1)
    selected_algorithm = allowed_algorithms[algorithm_index]

    # Create the Maze
    maze = mzl.Maze()
    maze.generator = selected_algorithm(10, 10)
    maze.generate()

    #
    maze.display_maze()
    print(maze.__str__())


if __name__ == '__main__':
    create_maze()
    print("Foobar")
