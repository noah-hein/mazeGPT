from transformers import TrainingArguments
from src.config.default import MazeAIConfig


class GpuConfig(MazeAIConfig):
    # ==================================================================================================================
    #       Maze Data
    # ==================================================================================================================

    NUMBER_OF_MAZES_PER_DIMENSION = 50000
    MIN_HEIGHT = 3
    MAX_HEIGHT = 6
    MIN_WIDTH = 3
    MAX_WIDTH = 6

    CHECKPOINT_MODEL = "checkpoint-180"
    USE_CHECKPOINT = True

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
            gradient_accumulation_steps=32,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            fp16=True,

            # gradient_checkpointing=True,
            save_total_limit=3,
            optim="adamw_torch",
    )
