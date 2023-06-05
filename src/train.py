import mazelib

if __name__ == '__main__':
    print("Foobar")

    maze = mazelib.Maze()
    maze.generator = mazelib.Prims(10, 10)
    maze.generate()

    maze.display_maze()
