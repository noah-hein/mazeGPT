import numpy as np
from src.maze.maze_algo import MazeAlgorithm


class AldousBroderAlgorithm(MazeAlgorithm):
    """
    Approach:
        The Aldous-Broder algorithm also produces uniform spanning trees.
        However, it is one of the least efficient maze algorithms.

    Algorithm:
        1. Pick a random cell as the current cell and mark it as visited.
        2. While there are unvisited cells:
            1. Pick a random neighbour.
            2. If the chosen neighbour has not been visited:
                1. Remove the wall between the current cell and the chosen neighbour.
                2. Mark the chosen neighbour as visited.
            3. Make the chosen neighbour the current cell.

    References:
        https://github.com/john-science/mazelib
        https://en.wikipedia.org/wiki/Maze_generation_algorithm#Aldous-Broder_algorithm
    """

    def __init__(self):
        super(AldousBroderAlgorithm, self).__init__()

    def generate(self):
        # Initialize empty grid
        grid = np.empty((self.blockHeight, self.blockWidth), dtype=np.int8)
        grid.fill(1)

        # Start with a random cell in the grid
        row, column = self._pick_random_cell()
        grid[row][column] = 0

        num_visited = 1
        while num_visited < self.height * self.width:
            neighbors = self._find_neighbors(row, column, grid, True)
            if not neighbors:
                row, column = self._pick_random_neighbor(row, column, grid)
            else:
                row, column, num_visited = self._process_neighbors(row, column, neighbors, grid, num_visited)
        return grid

    @staticmethod
    def _process_neighbors(row, column, neighbors, grid, num_visited):
        for nrow, ncol in neighbors:
            if grid[nrow, ncol] > 0:
                grid[(nrow + row) // 2, (ncol + column) // 2] = 0
                grid[nrow, ncol] = 0
                num_visited += 1
                row = nrow
                column = ncol
                break
        return row, column, num_visited

