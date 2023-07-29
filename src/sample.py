import random
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
        self.maze = Maze()
        self.maze.width = 5
        self.maze.height = 5
        self.maze.init_zero()
        max_length = self.maze.char_length() + 2

        # Create start tokens
        maze_start_sequence = "<|5x5|>"
        self.maze_string = maze_start_sequence

        # Animate the maze generation
        self.j = 0
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, self.update, frames=max_length, interval=100)
        plt.show()
        self.maze.display_maze()

    def update(self, frame):
        # Generate next token in sequence
        next_token = self.next_token(self.maze_string)
        self.maze_string += next_token
        print(self.maze_string)

        # Stop generating if the animation finishes
        if "<|end|>" in self.maze_string:
            plt.close()

        # Parse the next tokens and update maze
        characters = list(next_token)
        for c in characters:
            if c in ("0", "1"):
                row_index, col_index = np.unravel_index(self.j, self.maze.grid.shape)
                self.maze.grid[row_index][col_index] = c
                self.j = self.j + 1

                plt.imshow(self.maze.grid, cmap='binary', interpolation='none')
                plt.title(f"Frame {frame}")

    def next_token(self, maze_string):
        maze_string_length = len(maze_string)
        maze_string = self.generator(maze_string, max_new_tokens=1)[0]["generated_text"]
        next_token = maze_string[maze_string_length:]
        return next_token

