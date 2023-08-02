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
def detect(gross_path, bg_path, results_dir_path=None):
    import json

    import numpy as np

    from riid.anomaly import PoissonNChannelEventDetector

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
        long_term_duration=600,
        short_term_duration=1.5,
        pre_event_duration=5,
        max_event_duration=120,
        post_event_duration=1.5,
        tolerable_false_alarms_per_day=2,
        anomaly_threshold_update_interval=60,
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

    # detect_results = json.dumps(event_result_pairs)
    with open(f"{results_dir_path}/detect.json", "w+") as outfile:
        json.dump(event_result_pairs, outfile, indent=4)
        # outfile.write("{\n\t")
        # for i in event_result_pairs:
        #     outfile.write(json.dumps(i))
        #     outfile.write(': ')
        #     outfile.write(json.dumps(event_result_pairs[i]))
        #     if i == "measurement_ids":
        #         outfile.write('\n')
        #     else:
        #         outfile.write(', \n\t')
        # outfile.write("}")
    outfile.close

    """ TODO: Save results to JSON file with following structure:
    [
        {
            "gross_spectrum": event_result[0],
            "bg_spectrum": event_result[1],
            "duration_seconds": event_result[2],
            "measurement_ids": event_result[3]
        }
    ]
    """


@cli.command(short_help="Collect spectra from a device")
def sense():
    pass


cli.add_command(identify)
if __name__ == '__main__':
    cli()
