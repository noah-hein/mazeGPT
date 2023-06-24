from random import randint
from .maze import Maze
from .maze_gen_algo import MazeGenAlgo


class MazeFactory:
    # ==================================================================================================================
    #       Private Members
    # ==================================================================================================================

    MIN_SEED = 0
    MAX_SEED = 1000000

    # ==================================================================================================================
    #       Constructor
    # ==================================================================================================================

    def __init__(
            self,
            width: int,
            height: int,
            algorithm
    ):
        self.width = width
        self.height = height
        self.algorithm = algorithm

    # ==================================================================================================================
    #       Public Methods
    # ==================================================================================================================

    def generate_single(self, seed: int) -> Maze:
        """
        Creates a single new maze.
        These mazes are built upon the class private member maze parameters.
        """
        # Create the Maze
        algorithm = self.algorithm
        new_maze = Maze(seed)
        new_maze.generator = algorithm(self.height, self.width)
        new_maze.generate()
        return new_maze

    def generate_multiple(self, number_of_mazes: int):
        # Set up print table
        print("Generating mazes")
        table_header = "| {: >10} | {: >10} | {: >25} | {: >20} |".format("i", "dimensions", "algorithm", "seed")
        print(table_header)
        print(len(table_header) * "-")

        # Start generating the mazes
        mazes: list[Maze] = []
        for i in range(number_of_mazes):
            # Create new random maze
            seed = randint(self.MIN_SEED, self.MAX_SEED)
            new_maze = self.generate_single(seed)

            # Add and print maze
            mazes.append(new_maze)
            self._print_maze_info(i, 1)
        return mazes

    # ==================================================================================================================
    #       Private Methods
    # ==================================================================================================================

    def _print_maze_info(self, index: int, seed: int):
        maze_description = "| {: >10} | {: >10} | {: >25} | {: >20} |"
        maze_description = maze_description.format(
            index.__str__(),
            self.width.__str__() + "x" + self.height.__str__(),
            self.algorithm.__name__,
            seed
        )
        print(maze_description)
