import os

import click

from riid.data.sampleset import read_hdf
from riid.models.neural_nets import MLPClassifier

# @click.option('--verbose', is_flag=True, help="Show detailed output.")


@click.group(help="CLI tool for PyRIID")
def cli():
    pass


@cli.command(
    short_help="Train a pre-architected classifier or regressor on pre-synthesized gamma spectra")
def train():
    pass


@click.command(short_help="Identify measurements using a pre-trained classifier or regressor")
@click.argument('model_path', type=click.Path(exists=True, file_okay=True))
@click.argument('data_path', type=click.Path(exists=True, file_okay=True))
@click.option('--results_dir_path', '--results', metavar='',
              type=click.Path(exists=False, file_okay=True),
              help="Path to directory where identification results are output")
def identify(model_path, data_path, results_dir_path=None):

    print(f"Processing data with model: {model_path} and data: {data_path}")
    if not results_dir_path:
        results_dir_path = "./identify_results/"
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)

    model = MLPClassifier()
    model.load(model_path)
    data_ss = read_hdf(data_path)
    model.predict(data_ss)

    data_ss.prediction_probas.to_csv(results_dir_path + "results.csv")

    print("Done!")


@cli.command(
    short_help="Detect events within a series of gamma spectra based on a background measurement")
def detect():
    pass


@cli.command(short_help="Collect spectra from a device")
def sense():
    pass


cli.add_command(identify)
if __name__ == '__main__':
    cli()
