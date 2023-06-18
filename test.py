import os.path

from transformers import PreTrainedTokenizerFast, pipeline, GPT2LMHeadModel

import config
from mazelib import Maze, Prims

if __name__ == '__main__':
    # tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.TOKENIZER_FILE_PATH)
    #
    # model = GPT2LMHeadModel.from_pretrained(config.MODEL_PATH, local_files_only=True)
    #
    # generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    # test = generator("<start>", max_length=300)

    # print(test)

    # test_maze = Maze()
    # test_maze.generator = Prims(3, 3)
    # test_maze.generate()
    # test_maze.display_maze()
    # print(test_maze.grid)

    maze = Maze()
    maze.string_to_maze("<start> 1 1 1 1 1 1 1 2 1 0 0 0 0 0 1 2 1 0 1 0 1 0 1 2 1 0 1 0 0 0 1 2 1 0 1 1 1 0 1 2 1 0 0 0 0 0 1 2 1 1 1 1 1 1 1 2 <end>")
    maze.display_maze()

