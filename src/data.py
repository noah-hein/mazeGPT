import os
import pathlib
import shutil

from src.config import MazeAIConfig
from src.maze.maze import Maze
from src.maze.maze_factory import MazeFactory
from src.util import rooted


class MazeAIData:
    # ==================================================================================================================
    #       Constructor
    # ==================================================================================================================

    def __init__(self, config: MazeAIConfig):
        self.config = config
        self.maze_files: dict[str, list[Maze]] = {}
        self.maze_factory = MazeFactory()

    # ==================================================================================================================
    #       Public Methods
    # ==================================================================================================================

    def generate(self):
        self.clear_data_folder()
        self.generate_maze_files()
        self.save_maze_files()

    def clear_data_folder(self):
        data_path = rooted(self.config.output.data)
        if os.path.isdir(data_path):
            shutil.rmtree(data_path)
        os.makedirs(data_path, exist_ok=True)

    def generate_maze_files(self):
        config = self.config
        maze_factory = self.maze_factory
        maze_files = self.maze_files

        #
        maze_config = config.maze
        width = maze_config.width
        height = maze_config.height

        # Create mazes for every dimension
        print(range(height.min, height.max + 1))
        for w in range(width.min, width.max + 1):
            for h in range(height.min, height.max + 1):
                # Generate the binary_tree
                maze_factory.width = w
                maze_factory.height = h
                new_mazes = maze_factory.generate(maze_config.number_per_dimension)

                # Save maze to temporary dictionary
                maze_data_filename: str = w.__str__() + "x" + h.__str__() + ".txt"
                maze_files[maze_data_filename] = new_mazes

    def save_maze_files(self):
        print("Saving mazes to file(s)...")
        for maze_file_name, maze_list in self.maze_files.items():
            self.save_mazes_to_file(maze_file_name, maze_list)

    def save_mazes_to_file(self, filename: str, mazes: list[Maze]):
        # Create output for file
        data_directory = rooted(self.config.output.data)
        pathlib.Path(data_directory).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(data_directory, filename)

        # Delete previous file
        if os.path.isfile(file_path):
            print("DELETED: " + file_path)
            os.remove(file_path)

        # Write each maze to file
        print("SAVED: " + file_path)
        with open(file_path, 'w') as file:
            for maze in mazes:
                file.write(maze.__repr__() + "\n")

    def dimension_tokens(self):
        config = self.config
        tokens = []
        for width in range(config.maze.width.min, config.maze.width.max + 1):
            for height in range(config.maze.height.min, config.maze.height.max + 1):
                token = "[" + width.__str__() + "x" + height.__str__() + "]"
                tokens.append(token)
        return tokens


