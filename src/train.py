from config import MazeAIConfig
from trainer import MazeAITrainer


if __name__ == '__main__':
    config = MazeAIConfig()
    trainer = MazeAITrainer(config)
    trainer.train()
