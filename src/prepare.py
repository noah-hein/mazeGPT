import os
import pathlib
import config
import shutil

from maze import Maze, MazeFactory


class Prepare:
    # ==================================================================================================================
    #       Constructor
    # ==================================================================================================================

    def __init__(self):
        self.maze_files: dict[str, list[Maze]] = {}
        self.maze_factory = self.setup_maze_factory()

    # ==================================================================================================================
    #       Public Methods
    # ==================================================================================================================

    def setup_maze_factory(self):
        maze_factory = MazeFactory()
        maze_factory.algorithms = config.ALLOWED_ALGORITHMS
        maze_factory.print_header()
        return maze_factory

    def clear_data_folder(self):
        data_path = config.DATA_DIRECTORY
        shutil.rmtree(data_path)
        os.mkdir(data_path)

    def generate_maze_files(self):
        for width in range(config.MIN_WIDTH, config.MAX_WIDTH + 1):
            for height in range(config.MIN_HEIGHT, config.MAX_HEIGHT + 1):
                # Generate the mazes
                self.maze_factory.width = width
                self.maze_factory.height = height
                new_mazes = self.maze_factory.generate(config.NUMBER_OF_MAZES_PER_DIMENSION)

                # Save maze to temporary dictionary
                maze_data_filename: str = width.__str__() + "x" + height.__str__() + ".txt"
                self.maze_files[maze_data_filename] = new_mazes
                self.maze_factory.print_table_break()

    def save_maze_files(self):
        print("Saving mazes to file(s)...")
        for maze_file_name, maze_list in self.maze_files.items():
            self.save_mazes_to_file(maze_file_name, maze_list)

    def save_mazes_to_file(self, filename: str, mazes: list[Maze]):
        # Create output for file
        pathlib.Path(config.DATA_DIRECTORY).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(config.DATA_DIRECTORY, filename)

        # Delete previous file
        if os.path.isfile(file_path):
            print("DELETED: " + file_path)
            os.remove(file_path)

        # Write each maze to file
        print("SAVED: " + file_path)
        with open(file_path, 'w') as file:
            for maze in mazes:
                file.write(maze.__repr__() + "\n")


if __name__ == '__main__':
    prepare = Prepare()
    prepare.clear_data_folder()
    prepare.generate_maze_files()
    prepare.save_maze_files()

    # generated_mazes = generate_mazes()
    # build_dataset(generated_mazes)
    # build_tokenizer(generated_mazes)
