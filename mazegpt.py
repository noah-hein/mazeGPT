import click
from src.prepare import prepare as prepare_handler


@click.group()
def cli():
    pass


@click.command()
@click.option("--config", default="/src/config/base.py", help="The selected configuration class")
def prepare(config):
    prepare_handler(config)


cli.add_command(prepare)
if __name__ == '__main__':
    cli()
