import random
import config

from transformers import PreTrainedTokenizerFast, pipeline, GPT2LMHeadModel, set_seed
from maze import Maze

def find_max_length():
    """
    Determine how many characters to generate
    """
    width_length = (config.MAX_WIDTH * 2 + 1)
    height_length = (config.MAX_HEIGHT * 2 + 1)
    return width_length * height_length + height_length

if __name__ == '__main__':
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.TOKENIZER_FILE_PATH)

    model = GPT2LMHeadModel.from_pretrained(config.MODEL_PATH, local_files_only=True)

    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    set_seed(random.randint(0, 100000))

    # Determine how many characters to generate
    max_length = find_max_length()

    top_row_length = config.MAX_WIDTH * 2 - 2


    maze_string = generator("11111112", max_length=max_length)
    maze_string = maze_string[0]["generated_text"]
    maze_string = maze_string.replace(" ", "")

    print(maze_string)

    maze = Maze()
    maze.string_to_maze(maze_string)
    print(maze.grid)
    maze.display_maze()
