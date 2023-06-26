from datasets import load_dataset
from torch import cuda

from config import MazeAIConfig
from model import MazeAIModel
from tokenizer import MazeAITokenizer
from transformers import \
    PreTrainedTokenizerFast, \
    DataCollatorForLanguageModeling, \
    TrainingArguments, \
    Trainer


class MazeAITrainer(Trainer):

    def __init__(self):
        print("test")






def encode(unencoded_dataset):
    return tokenizer(
        unencoded_dataset["text"],
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True
    )

if __name__ == '__main__':
    # Choose the config
    config = MazeAIConfig()

    # Load data set and tokenizer
    dataset = load_dataset(config.DATA_DIRECTORY, split='train').train_test_split(test_size=0.1)

    # Load in the pretrained tokenizer
    tokenizer = MazeAITokenizer(config).load()

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
    model = MazeAIModel(config)
    trainer = Trainer(
        model=model,
        args=config.TRAINING_ARGS,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()
