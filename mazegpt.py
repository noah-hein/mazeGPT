import hydra
import colorama
import pyfiglet
from dotenv import load_dotenv

from src.sample import MazeAISampler
from src.prepare import MazeAIPrepare
from src.config import MazeAIConfig
from src.train import MazeAITrainer


@hydra.main(version_base=None, config_path="conf", config_name="default")
def maze_gpt(config: MazeAIConfig) -> None:
    # Bootstrap and print logo
    print_header()
    load_dotenv()
    colorama.init()

    # Select Action Script
    actions = {
        "prepare": MazeAIPrepare,
        "train": MazeAITrainer,
        "sample": MazeAISampler
    }
    actions[config.action](config)


def print_header():
    print(colorama.Fore.GREEN + pyfiglet.figlet_format("MazeGPT") + colorama.Style.RESET_ALL)


if __name__ == "__main__":
    maze_gpt()
