from src.mazelib import algorithms

SPECIAL_TOKENS = [
    "<start>",
    "<end>",
    "[PAD]",
    "[MASK]"
]

ALLOWED_ALGORITHMS = [
    algorithms.Prims,
    algorithms.AldousBroder,
    algorithms.BacktrackingGenerator,
    algorithms.BinaryTree,
]

NUMBER_OF_MAZES = 1000
TRAINING_PERCENT = 0.9

MIN_HEIGHT = 3
MAX_HEIGHT = 3
MIN_WIDTH = 3
MAX_WIDTH = 3

OUTPUT_DIRECTORY = "../data"
DATA_FILENAME = "dataset.txt"
TOKENIZER_FILENAME = "tokenizer.json"