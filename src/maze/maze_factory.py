import random
from tqdm import tqdm
from random import randint
from .algorithms.prims import PrimsAlgorithm
from .maze import Maze


class MazeFactory:
    # ==================================================================================================================
    #       Private Members
    # ==================================================================================================================

    MIN_SEED = 0
    MAX_SEED = 1000000000000000

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
        # Build progress bar
        dimension_string = self.width.__str__() + "x" + self.height.__str__()
        progress_bar = tqdm(range(number_of_mazes), desc=dimension_string, leave=False)

        # Start generating the binary_tree
        mazes: list[Maze] = []
        for _ in progress_bar:
            # Create new random maze
            seed = randint(self.MIN_SEED, self.MAX_SEED)
            algorithm = self._random_maze_algorthm()
            new_maze = self._generate_single(seed, algorithm)
            mazes.append(new_maze)
        return mazes

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
        new_maze = Maze(seed)
        new_maze.width = self.width
        new_maze.height = self.height
        new_maze.algorithm = algorithm()
        new_maze.generate()
        return new_maze
