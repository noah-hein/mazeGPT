import os
from src.maze.algorithms import AldousBroderAlgorithm, PrimsAlgorithm, BacktrackingAlgorithm, BinaryTreeAlgorithm


class MazeAIConfig:
    # ==================================================================================================================
    #       Maze Data
    # ==================================================================================================================

    ALLOWED_ALGORITHMS = [
        BinaryTreeAlgorithm,
        AldousBroderAlgorithm,
        PrimsAlgorithm,
        BacktrackingAlgorithm
    ]

    NUMBER_OF_MAZES_PER_DIMENSION = 100000
    TRAINING_PERCENT = 0.9

    MIN_HEIGHT = 5
    MAX_HEIGHT = 5
    MIN_WIDTH = 5
    MAX_WIDTH = 5

    # ==================================================================================================================
    #       Folder / File Names
    # ==================================================================================================================

    OUTPUT_DIRECTORY_NAME = "out"
    MODEL_DIRECTORY_NAME = "models"
    DATA_DIRECTORY_NAME = "data"

    DATA_FILENAME = "dataset.txt"
    TOKENIZER_FILENAME = "tokenizer.json"
    CHECKPOINT_MODEL = "checkpoint-7030"
    USE_CHECKPOINT = False

    # ==================================================================================================================
    #       Paths
    # ==================================================================================================================

    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    OUTPUT_PATH = os.path.join(ROOT_PATH, OUTPUT_DIRECTORY_NAME)

    MODEL_DIRECTORY = os.path.join(OUTPUT_PATH, MODEL_DIRECTORY_NAME)
    MODEL_PATH = os.path.join(MODEL_DIRECTORY, CHECKPOINT_MODEL)

    DATA_DIRECTORY = os.path.join(OUTPUT_PATH, DATA_DIRECTORY_NAME)
    TOKENIZER_FILE_PATH = os.path.join(OUTPUT_DIRECTORY_NAME, TOKENIZER_FILENAME)

    # ==================================================================================================================
    #       Tokenizer
    # ==================================================================================================================

    PAD_TOKEN = "[PAD]"
    MASK_TOKEN = "[MASK]"
    SPECIAL_TOKENS = [
        PAD_TOKEN,
        MASK_TOKEN
    ]
