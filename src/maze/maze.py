import random
import numpy as np
from matplotlib import pyplot as plt


class Maze:
    """


    References:
        https://github.com/john-science/mazelib
    """

    NEWLINE_CHARACTER = "2"

    # ==================================================================================================================
    #       Constructor
    # ==================================================================================================================

    def __init__(self, width=None, height=None, seed=None):
        self.grid = []
        self.algorithm = None
        self.width = width
        self.height = height
        self.randomize_seed(seed)

    # ==================================================================================================================
    #       Public Methods
    # ==================================================================================================================

    @staticmethod
    def randomize_seed(seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self):
        # Ensure an algorithm was selected
        if self.algorithm is None:
            raise Exception("Algorithm for new maze generation not selected")
        self.algorithm.set_dimensions(self.width, self.height)
        self.grid = self.algorithm.generate()

    def display_maze(self):
        plt.figure(figsize=(10, 5))
        plt.imshow(self.grid, cmap=plt.cm.binary, interpolation='nearest')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def char_length(self):
        width_len = 2 * self.width + 1
        height_len = 2 * (self.height + 1)
        return width_len * height_len


    def parse_string(self, string: str):
        result = "".join(filter(str.isnumeric, string))
        lines = result.split("2")
        grid = []
        for line in lines:
            nodes = list(map(int, list(line)))
            if len(nodes) > 0:
                grid.append(nodes)
        self.grid = np.array(grid)

    # ==================================================================================================================
    #       Class Methods
    # ==================================================================================================================

    def to_string(self):
        header = "[" + self.width.__str__() + "x" + self.height.__str__() + "]"
        footer = "[END]"
        string_rep = header + ""
        for row in self.grid:
            row_string = "".join(map(str, row)) + self.NEWLINE_CHARACTER
            string_rep += row_string
        return string_rep + footer

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

