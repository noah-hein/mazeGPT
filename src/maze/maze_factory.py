import random
from random import randint
from .algorithms.prims import PrimsAlgorithm
from .maze import Maze


class MazeFactory:
    # ==================================================================================================================
    #       Private Members
    # ==================================================================================================================

    MIN_SEED = 0
    MAX_SEED = 1000000

    TABLE_FORMAT = "| {: >10} | {: >10} | {: >25} | {: >20} |"
    TABLE_HEADER = TABLE_FORMAT.format("i", "dimensions", "algorithm", "seed")

    # ==================================================================================================================
    #       Constructor
    # ==================================================================================================================

    def __init__(self):
        self.width = 5
        self.height = 5
        self.algorithms = [PrimsAlgorithm]

    # ==================================================================================================================
    #       Public Methods
    # ==================================================================================================================

    def generate(self, number_of_mazes: int):
        # Start generating the binary_tree
        mazes: list[Maze] = []
        for i in range(number_of_mazes):
            # Create new random maze
            seed = randint(self.MIN_SEED, self.MAX_SEED)
            algorithm = self._random_maze_algorthm()
            new_maze = self._generate_single(seed, algorithm)

            # Add and print maze
            mazes.append(new_maze)
            self._print_maze_info(i, 1, algorithm)
        return mazes

    def print_table_break(self):
        print(len(self.TABLE_HEADER) * "-")

    def print_header(self):
        print(self.TABLE_HEADER)
        self.print_table_break()

    # ==================================================================================================================
    #       Private Methods
    # ==================================================================================================================

    def _random_maze_algorthm(self):
        algorithm_index = random.randint(0, len(self.algorithms) - 1)
        return self.algorithms[algorithm_index]

    def _generate_single(self, seed: int, algorithm) -> Maze:
        """
        Creates a single new maze.
        These binary_tree are built upon the class private member maze parameters.
        """
        # Create the Maze
        new_maze = Maze(seed)
        new_maze.generator = algorithm(self.height, self.width)
        new_maze.generate()
        return new_maze

    def _print_maze_info(self, index: int, seed: int, algorithm):
        maze_description = "| {: >10} | {: >10} | {: >25} | {: >20} |"
        maze_description = maze_description.format(
            index.__str__(),
            self.width.__str__() + "x" + self.height.__str__(),
            algorithm.__name__,
            seed
        )
        print(maze_description)
