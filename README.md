# Maze AI
Does some maze generation and stuff. Working on this because I'm bored.

## Abstract
There are plenty of maze algorithms already out there that do a decent job at generating perfect maze.
The problem with these algorithms is that even with noise and different seeds, recognizable patterns form.
The idea is to generate thousands of mazes with a variety of algorithms and make a transformer model learn them all.
By doing this the model will be able to make original maze incorporating a variety of algorithms.
This would hopefully take the best of all algorithms and mimic a more human like design pattern.

## Representing a Maze
The easiest approach to representing a maze is with graph theory!
For storage purposes we will represent the structure as a two-dimensional matrix.
Luckily the structure of a maze can be represented with a handful of digits.
The encoding will be baked into the maze datatype.

Encoding:
- 0 = empty space
- 1 = wall
- 2 = path
- 3 = newline
- 4 = start
- 5 = end

