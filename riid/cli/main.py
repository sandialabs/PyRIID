import os

import click

from riid.data.sampleset import read_hdf

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
@click.option('--results_dir_path', '--results', metavar='',
              type=click.Path(exists=False, file_okay=True),
              help="Path to directory where identification results are output")
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
def detect(gross_path, bg_path, results_dir_path=None, long_term_duration=None,
           short_term_duration=None, pre_event_duration=None, max_event_duration=None,
           post_event_duration=None, tolerable_false_alarms_per_day=None,
           anomaly_threshold_update_interval=None,
           event_gross_file_path=None, event_bg_file_path=None):

    import numpy as np

    from riid.anomaly import PoissonNChannelEventDetector
    from riid.data import SampleSet

    print(f"Detecting events with gross measurements:       {gross_path}")
    print(f"                 background measurements:       {bg_path}")
    if not results_dir_path:
        results_dir_path = "./detect_results/"
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)

    gross = read_hdf(gross_path)
    background = read_hdf(bg_path)

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

    event_result_keys = ["gross_spectrum", "bg_spectrum", "duration_seconds", "measurement_ids"]
    event_result_pairs = {}
    for key, value in zip(event_result_keys, event_result):
        if key == "measurement_ids":
            event_result_pairs[key] = list(value)
        else:
            event_result_pairs[key] = value.tolist()



@cli.command(short_help="Collect spectra from a device")
def sense():
    pass


cli.add_command(identify)
if __name__ == '__main__':
    cli()
