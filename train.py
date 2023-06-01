from src.maze import Maze
from src.generate.prims import Prims

if __name__ == '__main__':
    print("Foobar")

    maze = Maze()
    maze.generator = Prims(5, 5)
    maze.generate()

    maze.display_maze()
