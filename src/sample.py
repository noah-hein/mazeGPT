import random

from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, pipeline, set_seed
from src.config.default import MazeAIConfig
from src.maze.maze import Maze


def sample(config: MazeAIConfig):
    # Import tokenizer and model
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.tokenizer_path())
    model = GPT2LMHeadModel.from_pretrained(config.model_path(), local_files_only=True)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    set_seed(random.randint(0, 100000))

    # Set up the new maze
    maze = Maze()
    maze.width = 5
    maze.height = 5
    maze_length = maze.char_length()

    maze.parse_string("111111111112100000100012111110101012100010001012111010111012100000000012111111101112100000101012101110101012101000001012111111111112")
    maze.display_maze()

    # # Create maze tag
    # tag = "[" + maze.width.__str__() + "x" + maze.height.__str__() + "]"
    # max_len = len(tag) + maze_length
    # print(max_len)
    #
    # #
    # maze_string = generator(tag + "111111111112", max_length=max_len)
    # print(maze_string.__str__())
    # print(len(maze_string.__str__()))





if __name__ == '__main__':
    sample(MazeAIConfig())

    # # Import tokenizer and model
    # tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.TOKENIZER_FILE_PATH)
    # model = GPT2LMHeadModel.from_pretrained(config.MODEL_PATH, local_files_only=True)
    # generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    # set_seed(random.randint(0, 100000))
    #
    # # Set up the new maze
    # maze = Maze()
    # max_length = maze.find_max_length()
    # print(max_length)
    #
    #
    # maze_string = generator("11111112", max_length=max_length)
    # maze_string = maze_string[0]["generated_text"]
    # maze_string = maze_string.replace(" ", "")
    #
    # print(maze_string)
    #
    #
    #
    #
    # maze.string_to_maze(maze_string)
    # print(maze.grid)
    # maze.display_maze()
