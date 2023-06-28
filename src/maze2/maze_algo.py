from abc import abstractmethod
from random import randrange, choice
from numpy.random import shuffle


class MazeAlgorithm:
    width = 5
    height = 5
    blockWidth = 2 * width + 1
    blockHeight = 2 * height + 1

    # ==================================================================================================================
    #       Public Methods
    # ==================================================================================================================

    def set_dimensions(self, width, height):
        self.width = width
        self.height = height
        self.blockWidth = 2 * self.width + 1
        self.blockHeight = 2 * self.height + 1

    # ==================================================================================================================
    #       Abstract Methods
    # ==================================================================================================================

    @abstractmethod
    def generate(self):
        return None

    # ==================================================================================================================
    #       Private Methods
    # ==================================================================================================================

    def _pick_random_cell(self):
        row = randrange(1, self.blockHeight, 2)
        column = randrange(1, self.blockWidth, 2)
        return row, column

    def _pick_random_neighbor(self, row, column, grid):
        return choice(self._find_neighbors(row, column, grid))

    def _find_neighbors(self, r, c, grid, is_wall=False):
        ns = []
        if r > 1 and grid[r - 2][c] == is_wall:
            ns.append((r - 2, c))
        if r < self.blockHeight - 2 and grid[r + 2][c] == is_wall:
            ns.append((r + 2, c))
        if c > 1 and grid[r][c - 2] == is_wall:
            ns.append((r, c - 2))
        if c < self.blockWidth - 2 and grid[r][c + 2] == is_wall:
            ns.append((r, c + 2))
        shuffle(ns)
        return ns


