import click


@click.group()
def cli():
    pass


@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name', help='The person to greet.')
def hello(count, name):
    print("Hello World!")


cli.add_command(hello)
if __name__ == '__main__':
    cli()
