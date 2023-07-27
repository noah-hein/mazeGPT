import click
import pyfiglet
import colorama
from dotenv import load_dotenv

from src.config.default import MazeAIConfig
from src.config.available_configs import AVAILABLE_CONFIGS
from src.prepare import prepare as prepare_handler
from src.sample import sample as sample_handler
from src.train import MazeAITrainer


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """
    CLI for easily interacting with the different mazeGPT scripts.
    """
    if ctx.invoked_subcommand is None:  # Only Display logo for help command
        logo = colorama.Fore.GREEN + pyfiglet.figlet_format("MazeGPT") + colorama.Style.RESET_ALL
        click.echo(logo)
        click.echo(ctx.get_help())
    pass


@click.command(help="Builds a dataset of mazes and trains a tokenizer from it")
@click.option("--conf", default="default", help="The selected configuration class")
def prepare(config):
    prepare_handler(AVAILABLE_CONFIGS[config]())


@click.command(help="Starts training the selected model with from the dataset and tokenizer")
@click.option("--conf", default="default", help="The selected configuration class")
def train(config):
    MazeAITrainer(AVAILABLE_CONFIGS[config]())


@click.command(help="Builds an example maze with the provided model")
@click.option("--conf", default="default", help="The selected configuration class")
def sample(config):
    sample_handler(AVAILABLE_CONFIGS[config]())


cli.add_command(prepare)
cli.add_command(train)
cli.add_command(sample)


if __name__ == '__main__':
    load_dotenv()
    colorama.init()
    cli()
