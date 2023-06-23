from src import config
from model import model
from datasets import load_dataset
from torch import cuda
from transformers import \
    PreTrainedTokenizerFast, \
    DataCollatorForLanguageModeling, \
    TrainingArguments, \
    Trainer


def encode(unencoded_dataset):
    return tokenizer(
        unencoded_dataset["text"],
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True
    )


if __name__ == '__main__':
    # Load data set and tokenizer
    dataset = load_dataset(config.DATA_DIRECTORY, data_files=["dataset.txt"], split='train')
    dataset = dataset.train_test_split(test_size=0.1)

    # Load in the pretrained tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.TOKENIZER_FILE_PATH)
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"

    # Encode the datasets
    train_dataset = dataset["train"].map(encode, batched=True)
    test_dataset = dataset["test"].map(encode, batched=True)

    # Is GPU available?
    device = "cuda" if cuda.is_available() else "cpu"
    cuda.empty_cache()
    print("Using your (" + device + ") to train")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )
    training_args = TrainingArguments(
        output_dir=config.MODEL_DIRECTORY,
        evaluation_strategy="steps",
        overwrite_output_dir=True,
        num_train_epochs=10,
        save_steps=10,
        logging_steps=10,
        logging_strategy="steps",

        gradient_accumulation_steps=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        fp16=True,

        # gradient_checkpointing=True,
        save_total_limit=3,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()
