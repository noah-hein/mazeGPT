import os
import pathlib
import shutil

from src.config.default import MazeAIConfig
from src.maze.maze import Maze
from src.maze.maze_factory import MazeFactory


class MazeAIData:
    # ==================================================================================================================
    #       Constructor
    # ==================================================================================================================

    def __init__(self, config: MazeAIConfig):
        self.config = config
        self.maze_files: dict[str, list[Maze]] = {}
        self.maze_factory = self.setup_maze_factory()

    # ==================================================================================================================
    #       Public Methods
    # ==================================================================================================================

    def generate(self):
        self.clear_data_folder()
        self.generate_maze_files()
        self.save_maze_files()

    def setup_maze_factory(self):
        maze_factory = MazeFactory()
        maze_factory.algorithms = self.config.ALLOWED_ALGORITHMS
        return maze_factory

    def clear_data_folder(self):
        data_path = self.config.data_directory()
        if os.path.isdir(data_path):
            shutil.rmtree(data_path)
        os.makedirs(data_path, exist_ok=True)

    def generate_maze_files(self):
        config = self.config
        maze_factory = self.maze_factory
        maze_files = self.maze_files

        # Create mazes for every dimension
        for width in range(config.MIN_WIDTH, config.MAX_WIDTH + 1):
            for height in range(config.MIN_HEIGHT, config.MAX_HEIGHT + 1):
                # Generate the binary_tree
                maze_factory.width = width
                maze_factory.height = height
                new_mazes = maze_factory.generate(config.NUMBER_OF_MAZES_PER_DIMENSION)

                # Save maze to temporary dictionary
                maze_data_filename: str = width.__str__() + "x" + height.__str__() + ".txt"
                maze_files[maze_data_filename] = new_mazes

    def save_maze_files(self):
        print("Saving binary_tree to file(s)...")
        for maze_file_name, maze_list in self.maze_files.items():
            self.save_mazes_to_file(maze_file_name, maze_list)

    def save_mazes_to_file(self, filename: str, mazes: list[Maze]):
        config = self.config

        # Create output for file
        pathlib.Path(config.data_directory()).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(config.data_directory(), filename)

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
        for width in range(config.MIN_WIDTH, config.MAX_WIDTH + 1):
            for height in range(config.MIN_HEIGHT, config.MAX_HEIGHT + 1):
                token = "[" + width.__str__() + "x" + height.__str__() + "]"
                tokens.append(token)
        return tokens


