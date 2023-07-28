import random

from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, pipeline, set_seed

from src.config import MazeAIConfig
from src.maze.maze import Maze
from src.util import rooted


class MazeAISampler:

    def __init__(self, config: MazeAIConfig):
        # Import tokenizer and model
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=rooted(config.output.tokenizer))
        model = GPT2LMHeadModel.from_pretrained(rooted(config.model), local_files_only=True)
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        set_seed(random.randint(0, 100000))

        # Set up the new maze
        maze = Maze()
        maze.width = 5
        maze.height = 5
        max_length = maze.char_length() + 2

        #
        maze_start_sequence = "<|5x5|>"
        maze_string = maze_start_sequence

        for i in range(max_length):
            maze_string = generator(maze_string, max_new_tokens=1)[0]["generated_text"]
            print(maze_string)
            if "<|end|>" in maze_string:
                break
