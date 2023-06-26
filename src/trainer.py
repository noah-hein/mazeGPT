from datasets import load_dataset
from torch import cuda
from config import MazeAIConfig
from model import MazeAIModel
from tokenizer import MazeAITokenizer
from transformers import \
    DataCollatorForLanguageModeling, \
    Trainer


class MazeAITrainer(Trainer):

    def __init__(self, config: MazeAIConfig):
        self.config = config
        self.args = config.TRAINING_ARGS

        # Use CPU or GPU
        self.determine_device()

        # Set up trainer members
        self.model = MazeAIModel(config)
        self.tokenizer = MazeAITokenizer(config).load()
        self.dataset = load_dataset(config.DATA_DIRECTORY, split='train').train_test_split(test_size=0.1)
        self.train_dataset = self.dataset["train"].map(self.encode, batched=True)
        self.eval_dataset = self.dataset["test"].map(self.encode, batched=True)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.2
        )
        super().__init__()

    def encode(self, unencoded_dataset):
        return self.tokenizer(
            unencoded_dataset["text"],
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True
        )

    def determine_device(self):
        device = "cuda" if cuda.is_available() else "cpu"
        cuda.empty_cache()
        print("Using your (" + device + ") to train")