import click
from typing import Type
from src.config.default import MazeAIConfig
from src.prepare import prepare as prepare_handler
from src.train import train as train_handler

AVAILABLE_CONFIGS: dict[str, Type[MazeAIConfig]] = {
    "default": MazeAIConfig
}


@click.group()
def cli():
    pass


@click.command(help="Builds a dataset of mazes and trains a tokenizer from it")
@click.option("--config", default="default", help="The selected configuration class")
def prepare(config):
    prepare_handler(AVAILABLE_CONFIGS[config]())


@click.command(help="Starts training the selected model with from the dataset and tokenizer")
@click.option("--config", default="default", help="The selected configuration class")
def train(config):
    train_handler(AVAILABLE_CONFIGS[config]())


cli.add_command(prepare)
cli.add_command(train)


if __name__ == '__main__':
    cli()
