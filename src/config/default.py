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
    TEST_PERCENT = 0.1

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
    CHECKPOINT_MODEL = ""

    # ==================================================================================================================
    #       Tokenizer
    # ==================================================================================================================

    VOCAB_SIZE = 10
    FRAGMENT_LENGTH = 1000
    TOKENIZER_MIN_FREQUENCY = 300000
    BATCH_SIZE = 10000

    PAD_TOKEN = "[PAD]"
    MASK_TOKEN = "[MASK]"
    START_TOKEN = "3"
    END_TOKEN = "4"
    SPECIAL_TOKENS = [
        PAD_TOKEN,
        MASK_TOKEN,
        START_TOKEN,
        END_TOKEN
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

    def has_model(self) -> bool:
        return len(self.CHECKPOINT_MODEL) > 0

    def output_path(self) -> str:
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(root_path, self.OUTPUT_DIRECTORY_NAME)

    def model_directory(self) -> str:
        return os.path.join(self.output_path(), self.MODEL_DIRECTORY_NAME)

    def model_path(self) -> str:
        return os.path.join(self.model_directory(), self.CHECKPOINT_MODEL)

    def data_directory(self) -> str:
        return os.path.join(self.output_path(), self.DATA_DIRECTORY_NAME)

    def tokenizer_path(self) -> str:
        return os.path.join(self.OUTPUT_DIRECTORY_NAME, self.TOKENIZER_FILENAME)

