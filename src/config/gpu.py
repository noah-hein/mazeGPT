from transformers import TrainingArguments
from config.default import MazeAIConfig


class GpuConfig(MazeAIConfig):
    # ==================================================================================================================
    #       Maze Data
    # ==================================================================================================================

    NUMBER_OF_MAZES_PER_DIMENSION = 50000
    MIN_HEIGHT = 3
    MAX_HEIGHT = 6
    MIN_WIDTH = 3
    MAX_WIDTH = 6

    # ==================================================================================================================
    #       Training
    # ==================================================================================================================

    TRAINING_ARGS = TrainingArguments(
        output_dir=MazeAIConfig.MODEL_DIRECTORY,
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
