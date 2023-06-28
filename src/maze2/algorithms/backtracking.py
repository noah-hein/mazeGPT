import numpy as np
from maze2.maze_algo import MazeAlgorithm


class BacktrackingAlgorithm(MazeAlgorithm):
    """
    Approach:
        Form a recursive function, which will follow a path and check if the path reaches the destination or not.
        If the path does not reach the destination then backtrack and try other paths.

    Algorithm:
        1. Create a solution matrix, initially filled with 0’s.
        2. Create a recursive function, which takes initial matrix, output matrix and position of rat (i, j).
        3. If the position is out of the matrix or the position is not valid then return.
        4. Mark the position output[i][j] as 1 and check if the current position is destination or not.
           If destination is reached print the output matrix and return.
        5. Recursively call for position (i+1, j) and (i, j+1).
        6. Unmark position (i, j), i.e output[i][j] = 0.

    References:
        https://github.com/john-science/mazelib
        https://www.geeksforgeeks.org/rat-in-a-maze/#
    """

    def __init__(self):
        super(BacktrackingAlgorithm, self).__init__()

    def generate(self):
        grid = np.ones((self.blockHeight, self.blockWidth), dtype=np.int8)
        row, column = self._pick_random_cell()
        grid[row, column] = 0
        track = [(row, column)]
        while track:
            row, column = track[-1]
            neighbors = self._find_neighbors(row, column, grid, True)
            if not neighbors:
                track = track[:-1]
            else:
                new_row, new_column = neighbors[0]
                grid[new_row, new_column], grid[(new_row + row) // 2, (new_column + column) // 2] = 0, 0
                track.append((new_row, new_column))
        return grid
