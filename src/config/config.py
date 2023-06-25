import os
from maze.algorithms import *


class MazeAiConfig:
    # ==================================================================================================================
    #       Maze Settings
    # ==================================================================================================================

    ALLOWED_ALGORITHMS = [
        BinaryTree,
        AldousBroder,
        Prims,
        BacktrackingGenerator
    ]

    NUMBER_OF_MAZES_PER_DIMENSION = 300
    TRAINING_PERCENT = 0.9

    MIN_HEIGHT = 3
    MAX_HEIGHT = 4
    MIN_WIDTH = 3
    MAX_WIDTH = 4

    # ==================================================================================================================
    #       Tokenizer
    # ==================================================================================================================

    PAD_TOKEN = "[PAD]"
    MASK_TOKEN = "[MASK]"

    SPECIAL_TOKENS = [
        PAD_TOKEN,
        MASK_TOKEN
    ]

    # ==================================================================================================================
    #       Folder / File Names
    # ==================================================================================================================

    OUTPUT_DIRECTORY_NAME = "out"
    MODEL_DIRECTORY_NAME = "models"
    DATA_DIRECTORY_NAME = "data"

    DATA_FILENAME = "dataset.txt"
    TOKENIZER_FILENAME = "tokenizer.json"
    SELECTED_MODEL = "checkpoint-7030"

    # ==================================================================================================================
    #       Paths
    # ==================================================================================================================

    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    OUTPUT_PATH = os.path.join(ROOT_PATH, OUTPUT_DIRECTORY_NAME)

    MODEL_DIRECTORY = os.path.join(OUTPUT_PATH, MODEL_DIRECTORY_NAME)
    MODEL_PATH = os.path.join(MODEL_DIRECTORY, SELECTED_MODEL)

    DATA_DIRECTORY = os.path.join(OUTPUT_PATH, DATA_DIRECTORY_NAME)
    TOKENIZER_FILE_PATH = os.path.join(OUTPUT_DIRECTORY_NAME, TOKENIZER_FILENAME)
