import hydra

from src.prepare import prepare
from src.new_config import MazeAIConfig


@hydra.main(version_base=None, config_path="conf", config_name="default")
def my_app(config: MazeAIConfig) -> None:
    prepare(config)


if __name__ == "__main__":
    my_app()
