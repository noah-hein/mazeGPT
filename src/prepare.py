from datasets import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from src.new_config import MazeAIConfig
from src.data import MazeAIData
from src.util import bordered, rooted


def prepare(config: MazeAIConfig):
    """
    Builds dataset of mazes for training the model.
    Uses built data to then generate a tokenizer.
    Both dataset and tokenizer are saved to the output folder.
    """
    # Generate the training binary_tree
    print(bordered("Generating Mazes"))
    data = MazeAIData(config)
    data.generate()
    print()

    # Get training data for tokenizer
    print(bordered("Load Mazes For Training"))
    training_data = load_dataset(rooted(config.output.data))["train"]
    tokenizer = ByteLevelBPETokenizer()
    print()

    # Create iterator for moving through training data
    batch_size = config.tokenizer.batch_size
    batch_range = range(0, len(training_data), batch_size)
    batch_iterator = (training_data[i: i + batch_size]["text"] for i in batch_range)

    # Build special tokens list
    #dimension_tokens = data.dimension_tokens()
    #special_tokens = config.tokenizer + dimension_tokens

    # Train the tokenizer
    print(bordered("Training Tokenizer"))
    tokenizer.train_from_iterator(
        batch_iterator,
        show_progress=True,
        min_frequency=config.tokenizer.min_frequency,
        vocab_size=config.tokenizer.vocab_size,
    )

    # Save tokenizer to a file
    print("Saving tokenizer at " + config.output.tokenizer)
    tokenizer.save(rooted(config.output.tokenizer))

# if __name__ == '__main__':
#     prepare(MazeAIConfig)
