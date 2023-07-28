import random
from array import array

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, pipeline, set_seed
from src.config import MazeAIConfig
from src.maze.maze import Maze
from src.util import rooted


class MazeAISampler:

    def __init__(self, config: MazeAIConfig):
        # Import tokenizer and model
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=rooted(config.output.tokenizer))
        model = GPT2LMHeadModel.from_pretrained(rooted(config.model), local_files_only=True)
        self.generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        set_seed(random.randint(0, 100000))

        # Set up the new maze
        maze = Maze()
        maze.width = 5
        maze.height = 5
        maze.init_zero()
        max_length = maze.char_length() + 2

        #
        maze_start_sequence = "<|5x5|>"
        maze_string = maze_start_sequence

        j = 0
        for i in range(max_length):
            #
            next_token = self.next_token(maze_string)
            maze_string += next_token

            if "<|end|>" in maze_string:
                break

            characters = list(next_token)
            for c in characters:
                if c in ("0", "1"):
                    row_index, col_index = np.unravel_index(j, maze.grid.shape)
                    maze.grid[row_index][col_index] = c
                    j = j + 1

    def next_token(self, maze_string):
        maze_string_length = len(maze_string)
        maze_string = self.generator(maze_string, max_new_tokens=1)[0]["generated_text"]
        next_token = maze_string[maze_string_length:]
        return next_token

    def iterate_through_matrix(self, matrix, index):
        rows = len(matrix)
        cols = len(matrix[0])
        row = index // rows
        col = index % cols
        return row, col

    # def update_figure(self):
    #     return []
    #
    # def display_maze(self):
    #     fig = plt.figure()
    #     im = plt.imshow(arr[0], animated=True)
    #     ani = animation.FuncAnimation(fig, updatefig,  blit=True)

