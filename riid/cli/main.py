import os

import click

from riid.data.sampleset import read_hdf
from riid.data.sampleset import read_pcf
# from riid.cli.validate import validate_ext_is_supported
from pathlib import Path

SUPPORTED_FILE_TYPES = [".pcf", ".h5"]


def validate_ext_is_supported(file_path):
    path = Path(file_path)
    ext = path.suffix
    if ext not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"'{ext}' is an unsupported output file format.")

# @click.option('--verbose', is_flag=True, help="Show detailed output.")


@click.group(help="CLI tool for PyRIID")
def cli():
    pass


@cli.command(
    short_help="Train a pre-architected classifier or regressor on pre-synthesized gamma spectra")
@click.option('--model_type', type=click.Choice(['mlp', 'lpe', 'pb'], case_sensitive=False),
              required=True, help="Model type. Choices are: mlp, lpe, and pb")
@click.argument('data_path', type=click.Path(exists=True, file_okay=True))
@click.option('--model_path', type=click.Path(exists=True, file_okay=True))
@click.option('--result_dir_path', '--results', metavar='',
              type=click.Path(exists=True, file_okay=True),
              help="""Path to directory hwere training results are output including
                model info as a JSON file""")
def train(model_type, data_path, model_path=None, results_dir_path=None):

    print(f"Training model: {model_type} on data: {data_path}")
    if (model_type.casefold() == 'mlp'):
        pass
    elif (model_type.casefold() == 'lpe'):
        pass
    elif (model_type.casefold() == 'pb'):
        pass


@click.command(short_help="Identify measurements using a pre-trained classifier or regressor")
@click.argument('model_path', type=click.Path(exists=True, file_okay=True))
@click.argument('data_path', type=click.Path(exists=True, file_okay=True))
@click.option('--results_dir_path', '--results', metavar='',
              type=click.Path(exists=False, file_okay=True),
              help="Path to directory where identification results are output")
def identify(model_path, data_path, results_dir_path=None):
    from riid.models.neural_nets import MLPClassifier

    print(f"Identifying measurements with model:        {model_path}")
    print(f"                               data:        {data_path}")
    if not results_dir_path:
        results_dir_path = "./identify_results/"
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)

    # TODO: regenerate data files to test this command
    # TODO: schedule meeting with Tyler to review identify and detect commands before moving on

    model = MLPClassifier()
    model.load(model_path)
    data_ss = read_hdf(data_path)
    model.predict(data_ss)

    data_ss.prediction_probas.to_csv(results_dir_path + "results.csv")


@cli.command(
    short_help="Detect events within a series of gamma spectra based on a background measurement")
@click.argument('gross_path', type=click.Path(exists=True, file_okay=True))
@click.argument('bg_path', type=click.Path(exists=True, file_okay=True))
@click.option('--long_term_duration', metavar='', type=float, default=120.0, show_default=True,
              help="The duration (in seconds) of the long-term buffer")
@click.option('--short_term_duration', metavar='', type=float, default=1.0, show_default=True,
              help="The duration (in seconds) of the short-term buffer")
@click.option('--pre_event_duration', metavar='', type=float, default=5.0, show_default=True,
              help="The duration (in seconds) specifying the amount of pre-event, background"
              " samples to include in the event result")
@click.option('--max_event_duration', metavar='', type=float, default=120.0, show_default=True,
              help="The maximum duration (in seconds) of the event buffer")
@click.option('--post_event_duration', metavar='', type=float, default=1.5, show_default=True,
              help="The duration (in seconds) that determines the number of"
              " consecutive, insignificant measurements that must be observed"
              " in order to end an event")
@click.option('--tolerable_false_alarms_per_day', metavar='', type=int,
              default=1, show_default=True,
              help="The DESIRED maximum number of allowable false positive"
              " events per day for all spectrum channels")
@click.option('--anomaly_threshold_update_interval', metavar='', type=float,
              default=60.0, show_default=True,
              help="The time (in seconds) between updates to the anomaly probability thresholds")
@click.option('--event_gross_file_path', metavar='',
              type=click.Path(exists=False, file_okay=True),
              help="Path to file where gross event spectra are saved")
@click.option('--event_bg_file_path', metavar='',
              type=click.Path(exists=False, file_okay=True),
              help="Path to file where backgrounds for gross event spectra are saved")
def detect(gross_path, bg_path, long_term_duration=None,
           short_term_duration=None, pre_event_duration=None, max_event_duration=None,
           post_event_duration=None, tolerable_false_alarms_per_day=None,
           anomaly_threshold_update_interval=None,
           event_gross_file_path=None, event_bg_file_path=None):

    if not event_gross_file_path:
        path_gross = Path(gross_path)
        gross_results_path = Path(path_gross.parent, f"{path_gross.stem}_events{path_gross.suffix}")

    if not event_bg_file_path:
        path_bg = Path(bg_path)
        bg_results_path = Path(path_bg.parent, f"{path_bg.stem}_events{path_bg.suffix}")

    else:
        validate_ext_is_supported(event_gross_file_path)
        validate_ext_is_supported(event_bg_file_path)

        gross_results_path = Path(event_gross_file_path)
        bg_results_path = Path(event_bg_file_path)

    if gross_results_path.suffix != bg_results_path.suffix:
        raise ValueError("The desired format of the output file is ambiguous due to differing input"
                         " file types (PCF and HDF). Please provide an output_file_path.")

    import numpy as np
    import pandas as pd

    from riid.anomaly import PoissonNChannelEventDetector
    from riid.data import SampleSet

    print(f"Detecting events with gross measurements:       {gross_path}")
    print(f"                 background measurements:       {bg_path}")

    if path_gross.suffix == ".h5":
        gross = read_hdf(path_gross)
        background = read_hdf(path_bg)

    else:
        gross = read_pcf(path_gross)
        background = read_pcf(path_bg)

    bg_live_time = background.info.live_time.values[0]
    bg_cps = background.info.total_counts.values[0] / bg_live_time
    gross_live_time = gross.info.live_time.values[0]
    expected_bg_counts = gross_live_time * bg_cps
    expected_bg_measurement = background.spectra.iloc[0] * expected_bg_counts

    ed = PoissonNChannelEventDetector(
        long_term_duration,
        short_term_duration,
        pre_event_duration,
        max_event_duration,
        post_event_duration,
        tolerable_false_alarms_per_day,
        anomaly_threshold_update_interval,
    )

    print("Filling background...")
    measurement_id = 0
    while ed.background_percent_complete < 100:
        noisy_bg_measurement = np.random.poisson(expected_bg_measurement)
        _ = ed.add_measurement(
            measurement_id,
            noisy_bg_measurement,
            gross_live_time,
            verbose=False
        )
        measurement_id += 1

    events = []
    print("Detecting events...")
    if path_gross.suffix == ".h5":
        for i in range(gross.n_samples):
            gross_spectrum = gross.spectra.iloc[i].values
            event_result = ed.add_measurement(
                measurement_id,
                gross_spectrum,
                gross_live_time,
                verbose=False
            )
            measurement_id += 1
            if event_result:
                events.append(event_result)
    else:
        for i in range(gross.n_samples):
            gross_spectrum = gross.spectra.iloc[i].values
            event_result = ed.add_measurement(
                measurement_id,
                gross_spectrum.astype(float),
                gross_live_time.astype(float),
                verbose=False
            )
            measurement_id += 1.0
            if event_result:
                events.append(event_result)

    if ed.event_in_progress:
        print("Event still in progress, adding more backgrounds...")
        while not event_result:
            noisy_bg_measurement = np.random.poisson(expected_bg_measurement)
            event_result = ed.add_measurement(
                measurement_id,
                noisy_bg_measurement,
                gross_live_time,
                verbose=False
            )
            measurement_id += 1
            if event_result:
                events.append(event_result)

    num_events = len(events)
    events_msg_suffix = "" if num_events == 1 else "s"
    events_msg = f"{num_events} event{events_msg_suffix} detected."
    print(events_msg)

    for event_result in events:
        _, _, event_duration, measurement_ids = event_result
        first_measurement_id = measurement_ids[0]
        last_measurement_id = measurement_ids[-1]
        print(f"  > {event_duration:.2f}s from {first_measurement_id} to {last_measurement_id}")

    gross_ss = SampleSet()
    gross_ss.spectra = pd.DataFrame(event_result[0])
    gross_ss.info.live_time = event_result[2]
    gross_ss.info.first_measurement_id = float(event_result[3][0])
    gross_ss.info.last_measurement_id = event_result[3][-1]

    bg_ss = SampleSet()
    bg_ss.spectra = pd.DataFrame(event_result[1])
    bg_ss.info.live_time = event_result[2]
    bg_ss.info.first_measurement_id = float(event_result[3][0])
    bg_ss.info.last_measurement_id = event_result[3][-1]

    if gross_results_path.suffix == ".h5":
        gross_ss.to_hdf(str(gross_results_path))
    else:
        gross_ss.to_pcf(str(gross_results_path))

    if bg_results_path.suffix == ".h5":
        bg_ss.to_hdf(str(bg_results_path))
    else:
        bg_ss.to_pcf(str(bg_results_path))



@cli.command(short_help="Collect spectra from a device")
def sense():
    pass


cli.add_command(identify)
if __name__ == '__main__':
    cli()
