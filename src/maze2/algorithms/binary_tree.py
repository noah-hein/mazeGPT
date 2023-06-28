import numpy as np
from random import choice
from maze2.maze_algo import MazeAlgorithm


class BinaryTree(MazeAlgorithm):
    """
    Approach:
        A binary tree maze is a standard orthogonal maze where each cell always has a passage leading up or leading
        left, but never both. To create a binary tree maze, for each cell flip a coin to decide whether to add a
        passage leading up or left. Always pick the same direction for cells on the boundary, and the end result will
        be a valid simply connected maze that looks like a binary tree, with the upper left corner its root.

    References:
        https://github.com/john-science/mazelib
        https://en.wikipedia.org/wiki/Maze_generation_algorithm
    """

    def __init__(self):
        super(BinaryTree, self).__init__()
        skews = {
            "NW": [(1, 0), (0, -1)],
            "NE": [(1, 0), (0, 1)],
            "SW": [(-1, 0), (0, -1)],
            "SE": [(-1, 0), (0, 1)],
        }
        key = choice(list(skews.keys()))
        self.skew = skews[key]

    def generate(self):
        grid = np.empty((self.blockHeight, self.blockWidth), dtype=np.int8)
        grid.fill(1)
        for row in range(1, self.blockHeight, 2):
            for col in range(1, self.blockWidth, 2):
                grid[row][col] = 0
                neighbor_row, neighbor_col = self._find_neighbor(row, col)
                grid[neighbor_row][neighbor_col] = 0

        return grid

    def _find_neighbor(self, current_row, current_col):
        neighbors = []
        for b_row, b_col in self.skew:
            neighbor_row = current_row + b_row
            neighbor_col = current_col + b_col
            if 0 < neighbor_row < (self.blockHeight - 1):
                if 0 < neighbor_col < (self.blockWidth - 1):
                    neighbors.append((neighbor_row, neighbor_col))
        if len(neighbors) == 0:
            return current_row, current_col
        else:
            return choice(neighbors)
