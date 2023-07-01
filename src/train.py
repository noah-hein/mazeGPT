from src.config.default import MazeAIConfig
from src.trainer import MazeAITrainer


def train(config: MazeAIConfig):
    trainer = MazeAITrainer(config)
    trainer.train()


if __name__ == '__main__':
    """Allows you to run the train script without the CLI"""
    train(MazeAIConfig())
