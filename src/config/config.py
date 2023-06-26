import os

from transformers import TrainingArguments

from maze.algorithms import *


class MazeAIConfig:
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

    MIN_HEIGHT = 5
    MAX_HEIGHT = 5
    MIN_WIDTH = 5
    MAX_WIDTH = 5

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

    # ==================================================================================================================
    #       Training
    # ==================================================================================================================

    TRAINING_ARGS = TrainingArguments(
        output_dir=MODEL_DIRECTORY,
        evaluation_strategy="steps",
        overwrite_output_dir=True,
        num_train_epochs=10,
        save_steps=10,
        logging_steps=10,
        logging_strategy="steps",

        # learning_rate=5e-5,
        # weight_decay=0.1,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        fp16=True,

        # gradient_checkpointing=True,
        save_total_limit=3,
        optim="adamw_torch",
    )
