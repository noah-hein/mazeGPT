import hydra

from src.prepare import MazeAIPrepare
from src.new_config import MazeAIConfig, Action


@hydra.main(version_base=None, config_path="conf", config_name="default")
def maze_gpt(config: MazeAIConfig) -> None:
    print(config.action.name)
    if config.action is Action.INFO:
        print("farts")

    # MazeAIPrepare(config)


if __name__ == "__main__":
    maze_gpt()
