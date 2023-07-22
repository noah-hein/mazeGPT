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
@click.option("--config", default="default", help="The selected configuration class")
def prepare(config):
    prepare_handler(AVAILABLE_CONFIGS[config]())


@click.command(help="Starts training the selected model with from the dataset and tokenizer")
@click.option("--config", default="default", help="The selected configuration class")
def train(config):
    MazeAITrainer(AVAILABLE_CONFIGS[config]())


@click.command(help="Builds an example maze with the provided model")
@click.option("--config", default="default", help="The selected configuration class")
def sample(config):
    sample_handler(AVAILABLE_CONFIGS[config]())


# def commandWithConfigFile(config_file_param_name):
#
#     class CustomCommandClass(click.Command):
#
#         def invoke(self, ctx):
#             config_file = ctx.params[config_file_param_name]
#             if config_file is not None:
#
#                 config_data = AVAILABLE_CONFIGS[config_file]()
#                 for param, value in ctx.params.items():
#                     if value is None and param in config_data:
#                         ctx.params[param] = config_data[param]
#
#             return super(CustomCommandClass, self).invoke(ctx)
#
#     return CustomCommandClass


# @click.command(cls=commandWithConfigFile('config_file'))
# @click.argument("arg")
# @click.option("--opt")
# @click.option("--config", type=click.Path())
# def test(arg, opt, config_file):
#     print("arg: {}".format(arg))
#     print("opt: {}".format(opt))
#     print("config_file: {}".format(config_file))


cli.add_command(prepare)
cli.add_command(train)
cli.add_command(sample)


if __name__ == '__main__':
    load_dotenv()
    colorama.init()
    cli()
