from datasets import load_dataset
from src.config.default import MazeAIConfig
from src.util import determine_train_device
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast, DataCollatorForLanguageModeling, \
    TrainingArguments, Trainer


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
        self.TRAINING_ARGS = self._trainer_args()

        # Split the dataset
        dataset = load_dataset(config.DATA_DIRECTORY, split='train').train_test_split(test_size=0.1)
        print(dataset)
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
        self.train()

    # ==================================================================================================================
    #       Private Methods
    # ==================================================================================================================

    def _encode(self, unencoded_dataset):
        return self.TOKENIZER(
            unencoded_dataset["text"],
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True
        )

    def _model(self) -> GPT2LMHeadModel:
        checkpoint = self.config.CHECKPOINT
        has_checkpoint = self.config.CHECKPOINT is not None
        model = checkpoint if has_checkpoint else GPT2LMHeadModel(
                GPT2Config.from_pretrained(
                    "gpt2",
                    vocab_size=len(self.TOKENIZER),
                    bos_token_id=self.TOKENIZER.bos_token_id,
                    eos_token_id=self.TOKENIZER.eos_token_id,
                )
            )
        return model

    def _tokenizer(self) -> PreTrainedTokenizerFast:
        print(self.config.TOKENIZER_FILE_PATH)
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

    def _trainer_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.config.MODEL_DIRECTORY,
            evaluation_strategy="steps",
            overwrite_output_dir=True,
            num_train_epochs=10,
            save_steps=10,
            logging_steps=10,
            logging_strategy="steps",

            # learning_rate=5e-5,
            # weight_decay=0.1,
            gradient_accumulation_steps=8,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            fp16=False,

            # gradient_checkpointing=True,
            save_total_limit=3,
            optim="adamw_torch",
        )


if __name__ == '__main__':
    """Allows you to run the train script without the CLI"""
    MazeAITrainer(MazeAIConfig())
