from datasets import load_dataset
from src.config import MazeAIConfig
from src.util import determine_train_device, rooted
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast, DataCollatorForLanguageModeling, \
    Trainer, TrainingArguments


class MazeAITrainer(Trainer):
    # ==================================================================================================================
    #       Constructor
    # ==================================================================================================================

    def __init__(self, config: MazeAIConfig):
        # Use CPU or GPU
        determine_train_device()

        self.config = config
        self.TOKENIZER = self._tokenizer()
        self.DATA_COLLATOR = self._data_collator()
        self.MODEL = self._model()
        training_config = self.config.training.__dict__["_parent"]["training"]
        self.TRAINING_ARGS = TrainingArguments(**training_config)

        # Split the dataset
        data_dir = rooted(config.output.data)
        dataset = load_dataset(data_dir, split='train').train_test_split(test_size=config.maze.test_percent)
        train_dataset = dataset["train"].map(self._encode, batched=True)
        eval_dataset = dataset["test"].map(self._encode, batched=True)

        # Build the trainer
        super().__init__(
            args=self.TRAINING_ARGS,
            tokenizer=self.TOKENIZER,
            data_collator=self.DATA_COLLATOR,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model=self.MODEL
        )

        # Start training
        if self.has_model():
            self.train(rooted(config.output.model))
        else:
            self.train()

    # ==================================================================================================================
    #       Private Methods
    # ==================================================================================================================

    def _encode(self, unencoded_dataset):
        return self.TOKENIZER(
            unencoded_dataset["text"],
            max_length=self.config.tokenizer.fragment_length,
            return_special_tokens_mask=True,
            truncation=True
        )

    def _model(self) -> GPT2LMHeadModel:
        model_config = GPT2Config.from_pretrained(
            "gpt2",
            vocab_size=len(self.TOKENIZER),
            bos_token_id=self.TOKENIZER.bos_token_id,
            eos_token_id=self.TOKENIZER.eos_token_id,
        )
        if self.has_model():
            model_dir = rooted(self.config.output.model)
            model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_dir)
        return GPT2LMHeadModel(model_config)

    def _tokenizer(self) -> PreTrainedTokenizerFast:
        tokenizer_path = rooted(self.config.output.tokenizer)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        tokenizer.pad_token = self.config.tokenizer.pad_token
        tokenizer.mask_token = self.config.tokenizer.mask_token
        return tokenizer

    def _data_collator(self) -> DataCollatorForLanguageModeling:
        return DataCollatorForLanguageModeling(
            tokenizer=self.TOKENIZER,
            mlm=True,
            mlm_probability=0.2
        )

    def has_model(self):
        return len(self.config.model) > 0
