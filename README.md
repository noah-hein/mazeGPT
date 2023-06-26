# Maze AI
Does some maze generation and stuff. Working on this because I'm bored.

![Transformer](/media/transformer.jpg)

## Abstract
Recursive language models specifically transformers are very good at generating out semi-related strings.
The purpose of this experiment is to determine if this model could be applied to a more rigid two-dimensional
continuous structure.

## Representing a Maze
The easiest approach to representing a maze is with graph theory!
Each node in the graph can be thought of as a junction within the maze.

![Maze Graph](/media/maze_as_graph.png)

The focus of this project will be around perfect mazes. A perfect maze is the same as a spanning tree.
In fact several already existing algorithms use this principal for generation.

Perfect Maze Definition:
- No cycles
- No unfilled spaces (within the bounds)
- No matter where you start / end, there should only be one path

![Perfect vs Not Perfect Maze](/media/perfect_versus_not_perfect.png)

For storage purposes we will represent the structure as a two-dimensional matrix.
Each node in the maze (excluding the metadata nodes) can be represented as a 0 or 1.

## The Problem
There are plenty of maze algorithms already out there that do a decent job at generating perfect maze.
The problem with these algorithms is that even with noise and different seeds, recognizable patterns form.

|                      Prims                       |                            Binary Tree                             |
|:------------------------------------------------:|:------------------------------------------------------------------:|
| ![Prims Maze Example 1](/media/prims/prims1.png) | ![Binary Tree Maze Example 1](/media/binary_tree/binary_tree1.png) |
| ![Prims Maze Example 2](/media/prims/prims2.png) | ![Binary Tree Maze Example 2](/media/binary_tree/binary_tree2.png) |
| ![Prims Maze Example 3](/media/prims/prims3.png) | ![Binary Tree Maze Example 3](/media/binary_tree/binary_tree3.png) |

As you might begin to see, the used algorithms almost have a unique characteristic to them.
To the human eye these patterns become easily aparent at a distance. I've noticed this effect still holding
true no matter the size of the maze.

The idea would be to generate and train the network on thousands of mazes.
By doing this, hopefully the algorithm will learn how to make different segments and their relative relationships.
The end goal is to make a more human like design pattern, one without a fingerprint.

## Transformers?


## Tokenizer
So you might be asking "How the hell do you represent a graph with characters?"

For the sake of simplicity I've decided to go with an approach similar to binary (for now)
The encoding is as follows.

Encoding:
- 0 = path
- 1 = wall
- 2 = new line

In the future this could be expanded upon to allow for whitespace. This could potentially allow for
mazes of different shapes. I would also like to try and tackle a non box like three-dimensional maze 
(similar to a graph). Maybe somehow devise a way to spit out a vector representing the new nodes' location
or add another token to represent empty space, this way mazes of different shapes could be created.

![3D Graph](/media/3d_graph.png)

Since this is a very simple recurrent neural network, 
it operates in a linear fashion (Instead of in a higher dimension).
The maze can now be interpreted as a string of tokens, nice!

```math
begin{matrix} a & b \\ c & d \end{matrix}$
```



