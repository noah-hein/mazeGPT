# Maze AI
Does some maze generation and stuff. Working on this because I'm bored.

## Abstract
There are plenty of maze algorithms already out there that do a decent job at generating perfect maze.
The problem with these algorithms is that even with noise and different seeds, recognizable patterns form.
The idea is to generate thousands of mazes with a variety of algorithms and make a transformer model learn them all.
By doing this the model will be able to make original maze incorporating a variety of algorithms.
This would hopefully take the best of all algorithms and mimic a more human like design pattern.

## The Problem


## Representing a Maze
The easiest approach to representing a maze is with graph theory!
Each node in the graph can be thought of as a junction within the maze.
The focus of this project will be around perfect mazes. A perfect maze is the same as a spanning tree.
In fact several already existing algorithms use this principal for generation.

Perfect Maze Definition:
- No cycles
- All space is filled within th mazes bounds
- Only one path from start to finish




For storage purposes we will represent the structure as a two-dimensional matrix.
Each node in the maze (excluding the metadata nodes) can be represented as a 0 or 1.

### Encoding
Luckily the structure of a maze can be represented with a handful of digits.
The encoding will be baked into the maze datatype.

Encoding:
- 0 = path
- 1 = wall
- \n = newline
- \<start> = start of maze
- \<end> = end of maze

Since this is a very simple recurrent neural network, 
it operates via a linear fashion (Instead of in a higher dimension).
In the future it would be cool to somehow devise a way to spit out a vector representing the new nodes' location
or add another token to represent empty space, this way mazes of different shapes could be created.



