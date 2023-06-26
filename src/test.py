from maze import Maze
from maze.algorithms import *


if __name__ == '__main__':
    maze = Maze()
    maze.generator = AldousBroder(5, 5)
    maze.generate()

    print(maze.__repr__())
    print(maze.grid)
    maze.display_maze()

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
