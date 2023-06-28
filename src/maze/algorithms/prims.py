import numpy as np
from random import randrange
from maze.maze_algo import MazeAlgorithm


class PrimsAlgorithm(MazeAlgorithm):
    """
    Approach:
        The algorithm starts with an empty spanning tree. The idea is to maintain two sets of vertices. The first set
        contains the vertices already included in the MST, and the other set contains the vertices not yet included.
        At every step, it considers all the edges that connect the two sets and picks the minimum weight edge from these
        edges. After picking the edge, it moves the other endpoint of the edge to the set containing MST.

    Algorithm:
        1. Choose an arbitrary cell from the grid, and add it to some (initially empty) set visited nodes (V).
        2. Randomly select a wall from the grid that connects a cell in V with another cell not in V.
        3. Add that wall to the Minimal Spanning Tree (MST), and the edge's other cell to V.
        4. Repeat steps 2 and 3 until V includes every cell in G.

    References:
        https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
        https://en.wikipedia.org/wiki/Maze_generation_algorithm
    """

    def __init__(self):
        super(Prims, self).__init__()

    def generate(self):
        # create empty grid
        grid = np.empty((self.blockHeight, self.blockWidth), dtype=np.int8)
        grid.fill(1)

        # choose a random starting position
        current_row = randrange(1, self.blockHeight, 2)
        current_col = randrange(1, self.blockWidth, 2)
        grid[current_row][current_col] = 0

        # created a weighted list of all vertices connected in the graph
        neighbors = self._find_neighbors(current_row, current_col, grid, True)
        visited = 1
        while visited < self.height * self.width:
            nn = randrange(len(neighbors))
            current_row, current_col = neighbors[nn]
            visited += 1
            grid[current_row][current_col] = 0
            neighbors = neighbors[:nn] + neighbors[nn + 1:]
            nearest_n0, nearest_n1 = self._find_neighbors(
                current_row, current_col, grid
            )[0]
            grid[(current_row + nearest_n0) // 2][(current_col + nearest_n1) // 2] = 0
            unvisited = self._find_neighbors(current_row, current_col, grid, True)
            neighbors = list(set(neighbors + unvisited))
        return grid
