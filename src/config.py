import os
from dataclasses import dataclass

from transformers import TrainingArguments
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

    NUMBER_OF_MAZES_PER_DIMENSION: int = 100000
    TEST_PERCENT: float = 0.1

    MIN_HEIGHT: int = 3
    MAX_HEIGHT: int = 3
    MIN_WIDTH: int = 3
    MAX_WIDTH: int = 3

    # ==================================================================================================================
    #       Folder / File Names
    # ==================================================================================================================

    OUTPUT_DIRECTORY_NAME = "out"
    MODEL_DIRECTORY_NAME = "models"
    DATA_DIRECTORY_NAME = "data"

    DATA_FILENAME = "dataset.txt"
    TOKENIZER_FILENAME = "tokenizer.json"
    CHECKPOINT_MODEL = ""

    # ==================================================================================================================
    #       Tokenizer
    # ==================================================================================================================

    VOCAB_SIZE = 1000
    FRAGMENT_LENGTH = 1000
    TOKENIZER_MIN_FREQUENCY = 100000
    BATCH_SIZE = 10000

    PAD_TOKEN = "[PAD]"
    MASK_TOKEN = "[MASK]"
    SPECIAL_TOKENS = [
        PAD_TOKEN,
        MASK_TOKEN
    ]

    # ==================================================================================================================
    #       Public Methods
    # ==================================================================================================================

    def training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.model_directory(),
            evaluation_strategy="steps",
            overwrite_output_dir=True,
            num_train_epochs=10,
            save_steps=10,
            logging_steps=10,
            logging_strategy="steps",

            # learning_rate=5e-5,
            # weight_decay=0.1,
            gradient_accumulation_steps=32,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            fp16=False,

            # gradient_checkpointing=True,
            save_total_limit=3,
            optim="adamw_torch",
        )