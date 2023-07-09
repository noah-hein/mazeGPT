import click
import pyfiglet
import colorama

from src.config.available_configs import AVAILABLE_CONFIGS
from src.prepare import prepare as prepare_handler
from src.train import MazeAITrainer


@click.group()
def cli():
    """
    CLI for easily interacting with the different mazeGPT scripts.
    """
    pass


@click.command(help="Builds a dataset of mazes and trains a tokenizer from it")
@click.option("--config", default="default", help="The selected configuration class")
def prepare(config):
    prepare_handler(AVAILABLE_CONFIGS[config]())


@click.command(help="Starts training the selected model with from the dataset and tokenizer")
@click.option("--config", default="default", help="The selected configuration class")
def train(config):
    MazeAITrainer(AVAILABLE_CONFIGS[config]())


cli.add_command(prepare)
cli.add_command(train)


if __name__ == '__main__':
    click.clear()
    colorama.init()
    logo = pyfiglet.figlet_format("MazeGPT")
    print(colorama.Fore.GREEN, logo, colorama.Style.RESET_ALL)
    cli()
