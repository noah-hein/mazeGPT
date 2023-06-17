from datasets import load_dataset
from transformers import \
    PreTrainedTokenizerFast, \
    BertConfig, \
    BertForMaskedLM, \
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
    dataset = load_dataset("../data", data_files=["dataset.txt"], split='train')
    dataset = dataset.train_test_split(test_size=0.1)

    # Load in the pretrained tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="../data/tokenizer.json")
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"

    train_dataset = dataset["train"].map(encode, batched=True)
    test_dataset = dataset["test"].map(encode, batched=True)

    model_config = BertConfig()
    model = BertForMaskedLM(config=model_config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )

    training_args = TrainingArguments(
        output_dir="../temp",  # output directory to where save model checkpoint
        evaluation_strategy="steps",  # evaluate each `logging_steps` steps
        overwrite_output_dir=True,
        num_train_epochs=10,  # number of training epochs, feel free to tweak
        per_device_train_batch_size=10,  # the training batch size, put it as high as your GPU memory fits
        gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
        per_device_eval_batch_size=64,  # evaluation batch size
        logging_steps=1000,  # evaluate, log and save model checkpoints every 1000 step
        save_steps=1000,
        optim="adamw_torch"
        # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
        # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
