from dataclasses import dataclass, field
from enum import Enum

from transformers import TrainingArguments


class Action(Enum):
    INFO = "info"
    PREPARE = "prepare"
    TRAIN = "train"
    SAMPLE = "sample"


@dataclass
class DimensionConfig:
    min: int = 3
    max: int = 3


@dataclass
class MazeConfig:
    height: DimensionConfig = field(default_factory=DimensionConfig)
    width: DimensionConfig = field(default_factory=DimensionConfig)
    test_percent: float = 0.1
    number_per_dimension: int = 100000


@dataclass
class OutputConfig:
    dir: str = "out"
    model: str = "out/models"
    data: str = "out/data"
    tokenizer: str = "out/tokenizer.json"


@dataclass
class TokenizerConfig:
    vocab_size: int = 1000
    fragment_length: int = 1000
    min_frequency: int = 100000
    batch_size: int = 10000
    pad_token: str = "[PAD]"
    mask_token: str = "[MASK]"


@dataclass
class SampleConfig:
    width: int = 5
    height: int = 5


class TrainingConfig(TrainingArguments):
    output_dir = "out/models"
    evaluation_strategy = "steps"
    overwrite_output_dir = True
    num_train_epochs = 10
    save_steps = 10
    logging_steps = 10
    logging_strategy = "steps"
    gradient_accumulation_steps = 32
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 16
    fp16 = False
    save_total_limit = 3
    optim = "adamw_torch"


@dataclass
class MazeAIConfig:
    action: Action = Action.INFO
    model: str = ""
    maze: MazeConfig = field(default_factory=MazeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingArguments = field(default_factory=TrainingConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
