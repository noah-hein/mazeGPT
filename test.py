import os.path
import random

from transformers import PreTrainedTokenizerFast, pipeline, GPT2LMHeadModel, set_seed

import config
from mazelib import Maze, Prims

if __name__ == '__main__':
    # tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.TOKENIZER_FILE_PATH)
    #
    # model = GPT2LMHeadModel.from_pretrained(config.MODEL_PATH, local_files_only=True)
    #
    # generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    # set_seed(random.randint(0, 100000))
    # test = generator(
    #     "<start>",
    #     max_length=200
    #     # num_beams=5,
    #     # no_repeat_ngram_size=2,
    #     # num_return_sequences=5,
    #     # early_stopping=True
    # )
    #
    # print(test)

    maze = Maze()
    maze.string_to_maze("<start> 1 1 1 1 1 1 1 2 1 0 0 0 0 0 1 2 1 0 1 0 1 0 1 2 1 0 0 0 1 0 1 2 1 0 1 0 1 0 1 2 1 0 0 0 0 0 1 2 1 1 1 1 1 1 1 2 <stop>")
    maze.display_maze()

