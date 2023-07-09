from datasets import load_dataset
from src.config.default import MazeAIConfig
from src.util import determine_train_device
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast, DataCollatorForLanguageModeling, Trainer


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
        self.TRAINING_ARGS = self.config.TRAINING_ARGS

        # Split the dataset
        dataset = load_dataset(config.DATA_DIRECTORY, split='train').train_test_split(test_size=config.TEST_SIZE)
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
        if config.USE_CHECKPOINT:
            self.train(self.config.MODEL_PATH)
        else:
            self.train()

    # ==================================================================================================================
    #       Private Methods
    # ==================================================================================================================

    def _encode(self, unencoded_dataset):
        return self.TOKENIZER(
            unencoded_dataset["text"],
            max_length=self.config.FRAGMENT_LENGTH,
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
        if self.config.USE_CHECKPOINT:
            model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=self.config.MODEL_PATH)
        return GPT2LMHeadModel(model_config)

    def _tokenizer(self) -> PreTrainedTokenizerFast:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.config.TOKENIZER_FILE_PATH)
        tokenizer.pad_token = self.config.PAD_TOKEN
        tokenizer.mask_token = self.config.MASK_TOKEN
        return tokenizer

    def _data_collator(self) -> DataCollatorForLanguageModeling:
        return DataCollatorForLanguageModeling(
            tokenizer=self.TOKENIZER,
            mlm=True,
            mlm_probability=0.2
        )


if __name__ == '__main__':
    """Allows you to run the train script without the CLI"""
    MazeAITrainer(MazeAIConfig())
