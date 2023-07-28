import hydra

from src.prepare import MazeAIPrepare
from src.new_config import MazeAIConfig
from src.train import MazeAITrainer


@hydra.main(version_base=None, config_path="conf", config_name="default")
def maze_gpt(config: MazeAIConfig) -> None:
    actions = {
        "prepare": MazeAIPrepare,
        "train": MazeAITrainer
    }
    actions[config.action](config)


if __name__ == "__main__":
    maze_gpt()
