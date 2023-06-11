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
- 0 = path
- 1 = wall
- \n = newline
- \<start> = newline
- \<end> = start

Below is the acutal array storing the tokens and their position
```python
TOKENS = [
    "0",
    "1",
    "\n",
    "<start>",
    "<end>"
]
```

## TODO (Priority)
- [x] Obtain maze generation algorithms
- [x] Determine shape of maze data
- [x] Create encoder and decoder
- [ ] Generate a bunch of training and validation mazes
- [ ] Place training and validation mazes into bin files
- [ ] Create transformer model
- [ ] Build train script for training the transformer
- [ ] Store model as a file for reuse
- [ ] Test blank maze generation with start token

## TODO (Later)
- [ ] Add CUDA support
- [ ] Create animated graphic to visualize maze generation
- [ ] Improve documentation / Add images



