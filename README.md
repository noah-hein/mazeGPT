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

## TODO (Priority)
- [x] Obtain maze generation algorithms
- [x] Determine shape of maze data
- [x] Create encoder and decoder
- [x] Generate a bunch of training and validation mazes
- [x] Place training and validation mazes into bin files
- [ ] Create transformer model
- [ ] Build train script for training the transformer
- [ ] Store model as a file for reuse
- [ ] Test blank maze generation with start token

## TODO (Later)
- [ ] Add CUDA support
- [ ] Create animated graphic to visualize maze generation
- [ ] Improve documentation / Add images



