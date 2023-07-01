import click
from typing import Type
from src.config.default import MazeAIConfig
from src.prepare import prepare as prepare_handler


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


cli.add_command(prepare)
if __name__ == '__main__':
    cli()
