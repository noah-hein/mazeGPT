from datasets import load_dataset
from pynvml import *
from torch import cuda
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


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


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

    device = "cuda" if cuda.is_available() else "cpu"
    cuda.empty_cache()
    print("Using your (" + device + ") to train")

    model_config = BertConfig()
    model = BertForMaskedLM(config=model_config).to(device)

    training_args = TrainingArguments(
        output_dir="../temp",
        evaluation_strategy="steps",
        overwrite_output_dir=True,
        num_train_epochs=10,
        save_steps=10,
        logging_steps=10,
        logging_strategy="steps",

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=32,

        optim="adamw_torch",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()
