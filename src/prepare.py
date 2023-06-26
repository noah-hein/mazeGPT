from config import MazeAIConfig
from data import MazeAIData
from tokenizer import MazeAiTokenizer

if __name__ == '__main__':
    # Choose the config
    config = MazeAIConfig()

    # Generate the training mazes
    data = MazeAIData(config)
    data.generate()

    # Build the tokenizer based on data
    tokenizer = MazeAiTokenizer(config)
    tokenizer.train()
