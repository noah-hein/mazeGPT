import os
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

    NUMBER_OF_MAZES_PER_DIMENSION = 100000
    TRAINING_PERCENT = 0.9

    MIN_HEIGHT = 3
    MAX_HEIGHT = 3
    MIN_WIDTH = 3
    MAX_WIDTH = 3

    # ==================================================================================================================
    #       Folder / File Names
    # ==================================================================================================================

    OUTPUT_DIRECTORY_NAME = "out"
    MODEL_DIRECTORY_NAME = "models"
    DATA_DIRECTORY_NAME = "data"

    DATA_FILENAME = "dataset.txt"
    TOKENIZER_FILENAME = "tokenizer.json"
    CHECKPOINT_MODEL = "checkpoint-180"
    USE_MODEL = False
    RESUME_FROM_CHECKPOINT = False

    # ==================================================================================================================
    #       Tokenizer
    # ==================================================================================================================

    VOCAB_SIZE = 10
    PAD_TOKEN = "[PAD]"
    MASK_TOKEN = "[MASK]"
    SPECIAL_TOKENS = [
        PAD_TOKEN,
        MASK_TOKEN
    ]

    # ==================================================================================================================
    #       Training
    # ==================================================================================================================

    TEST_SIZE = 0.1
    FRAGMENT_LENGTH = 1000

    def training_args(self):
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

    # ==================================================================================================================
    #       Public Methods
    # ==================================================================================================================

    def output_path(self):
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(root_path, self.OUTPUT_DIRECTORY_NAME)

    def model_directory(self):
        return os.path.join(self.output_path(), self.MODEL_DIRECTORY_NAME)

    def model_path(self):
        return os.path.join(self.model_directory(), self.CHECKPOINT_MODEL)

    def data_directory(self):
        return os.path.join(self.output_path(), self.DATA_DIRECTORY_NAME)

    def tokenizer_path(self):
        return os.path.join(self.OUTPUT_DIRECTORY_NAME, self.TOKENIZER_FILENAME)

