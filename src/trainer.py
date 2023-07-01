from datasets import load_dataset
from src.config.default import MazeAIConfig
from src.model import MazeAIModel
from src.tokenizer import MazeAITokenizer
from src.util import determine_train_device
from transformers import \
    DataCollatorForLanguageModeling, \
    Trainer


class MazeAITrainer(Trainer):

    def __init__(self, config: MazeAIConfig):
        self.config = config
        self.args = config.TRAINING_ARGS

        # Use CPU or GPU
        determine_train_device()

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

        # Pass arguments to trainer class
        super().__init__(
            args=self.args,
            tokenizer=self.tokenizer,
            model=self.model,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )

    def encode(self, unencoded_dataset):
        return self.tokenizer(
            unencoded_dataset["text"],
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True
        )
