import click

@click.option('--verbose', is_flag=True, help="Show detailed output.")

@click.group(help="CLI tool for PyRIID")
def cli():
    pass

@cli.command(short_help="Train a pre-architected classifier or regressor on pre-synthesized gamma spectra")
def train():
    pass

@cli.command(short_help="Identify measurements using a pre-trained classifier or regressor")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option('--results_dir_path', '--results', metavar='', help="Path to directory where identification results are output")
def identify(model_path, data_path):
    pass

@cli.command(short_help="Detect events within a series of gamma spectra based on a background measurement")
def detect():
    pass

if __name__ == '__main__':
    cli()