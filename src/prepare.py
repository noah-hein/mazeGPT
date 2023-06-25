from config import MazeAiConfig
from data import MazeAiData
from tokenizer import MazeAiTokenizer

if __name__ == '__main__':
    # Choose the config
    config = MazeAiConfig()

    # Generate the training mazes
    data = MazeAiData(config)
    data.generate()

    # Build the tokenizer based on data
    tokenizer = MazeAiTokenizer(config)
    tokenizer.train()
