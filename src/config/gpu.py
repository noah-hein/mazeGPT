from transformers import TrainingArguments
from src.config.default import MazeAIConfig


class GpuConfig(MazeAIConfig):
    # ==================================================================================================================
    #       Maze Data
    # ==================================================================================================================

    NUMBER_OF_MAZES_PER_DIMENSION = 500000
    MIN_HEIGHT = 5
    MAX_HEIGHT = 5
    MIN_WIDTH = 5
    MAX_WIDTH = 5

    CHECKPOINT_MODEL = "checkpoint-1900"
    USE_MODEL = False

    # ==================================================================================================================
    #       Tokenizer
    # ==================================================================================================================

    VOCAB_SIZE = 1000
    FRAGMENT_LENGTH = 1000
    TOKENIZER_MIN_FREQUENCY = 500000
    BATCH_SIZE = 10000

    # ==================================================================================================================
    #       Training
    # ==================================================================================================================

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
            gradient_accumulation_steps=1,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=40,
            fp16=True,

            # gradient_checkpointing=True,
            save_total_limit=3,
            optim="adamw_torch"
        )

