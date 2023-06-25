from config import MazeAiConfig
from data import MazeAiData

if __name__ == '__main__':
    config = MazeAiConfig()
    data = MazeAiData(config)
    data.generate()

    # generated_mazes = generate_mazes()
    # build_dataset(generated_mazes)
    # build_tokenizer(generated_mazes)
