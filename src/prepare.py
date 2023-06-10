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


def create_maze():
    # Select a random algorithm
    algorithm_index = random.randint(0, len(allowed_algorithms) - 1)
    selected_algorithm = allowed_algorithms[algorithm_index]

    # Create the Maze
    maze = mzl.Maze()
    maze.generator = selected_algorithm(10, 10)
    maze.generate()
    maze.display_maze()


if __name__ == '__main__':
    create_maze()
    print("Foobar")
