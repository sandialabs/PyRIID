# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to obtain events using an anomaly detection
algorithm.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from riid.anomaly import PoissonNChannelEventDetector
from riid.data.labeling import BACKGROUND_LABEL
from riid.data.synthetic.passby import PassbySynthesizer
from riid.data.synthetic.static import get_dummy_sampleset

SAMPLE_INTERVAL = 0.5
BG_RATE = 300
EXPECTED_BG_COUNTS = SAMPLE_INTERVAL * BG_RATE
SHORT_TERM_DURATION = 1.5
POST_EVENT_DURATION = 1.5
N_POST_EVENT_SAMPLES = (POST_EVENT_DURATION + SAMPLE_INTERVAL) / SAMPLE_INTERVAL

seed_ss = get_dummy_sampleset(as_seeds=True)

bg_seed_ss = seed_ss[seed_ss.get_labels() == BACKGROUND_LABEL]
bg_seed_ss.to_pmf()
ed = PoissonNChannelEventDetector(
    long_term_duration=600,
    short_term_duration=SHORT_TERM_DURATION,
    pre_event_duration=5,
    max_event_duration=120,
    post_event_duration=POST_EVENT_DURATION,
    tolerable_false_alarms_per_day=2,
    anomaly_threshold_update_interval=60,
)
count_rate_history = []

# Fill background buffer first
print("Filling background")
measurement_id = 0
expected_bg_measurement = bg_seed_ss.spectra.iloc[0] * EXPECTED_BG_COUNTS
while ed.background_percent_complete < 100:
    noisy_bg_measurement = np.random.poisson(expected_bg_measurement)
    count_rate = noisy_bg_measurement.sum() / SAMPLE_INTERVAL
    count_rate_history.append(count_rate)
    _ = ed.add_measurement(
        measurement_id,
        noisy_bg_measurement,
        SAMPLE_INTERVAL
    )
    measurement_id += 1

# Now let's see how many false alarms we get.
# You may want to make this duration higher in order to get a better statistic.
print("Checking false alarm rate")
FALSE_ALARM_CHECK_DURATION = 3 * 60 * 60
false_alarm_check_range = range(
    measurement_id,
    int(FALSE_ALARM_CHECK_DURATION / SAMPLE_INTERVAL) + 1
)
false_alarms = 0
for measurement_id in false_alarm_check_range:
    noisy_bg_measurement = np.random.poisson(expected_bg_measurement)
    count_rate = noisy_bg_measurement.sum() / SAMPLE_INTERVAL
    count_rate_history.append(count_rate)
    event_result = ed.add_measurement(
        measurement_id,
        noisy_bg_measurement,
        SAMPLE_INTERVAL
    )
    if event_result:
        false_alarms += 1
    measurement_id += 1
false_alarm_rate = 60 * 60 * false_alarms / FALSE_ALARM_CHECK_DURATION
print(f"False alarm rate: {false_alarm_rate:.2f}/hour")

# Now let's make a single passby
print("Generating pass-by")
passby_ss = PassbySynthesizer(
    seeds=seed_ss,
    sample_interval=SAMPLE_INTERVAL,
    background_cps=BG_RATE,
    subtract_background=True,
)._generate_single_passby(
    fwhm=1,
    snr=0.5,
    dwell_time=1,
    seed_pdf=seed_ss.spectra.iloc[0].values,
    background_pdf=bg_seed_ss.spectra.iloc[0].values,
    source=seed_ss.get_labels()[0]
)

print("Passing by...")
passby_range = list(range(measurement_id, passby_ss.n_samples + measurement_id))
starting_measurement_id = measurement_id
for i, measurement_id in enumerate(passby_range):
    fg_spectrum = passby_ss.spectra.iloc[i].values
    noisy_bg_measurement = np.random.poisson(expected_bg_measurement)
    gross_spectrum = fg_spectrum + noisy_bg_measurement
    count_rate = gross_spectrum.sum() / SAMPLE_INTERVAL
    count_rate_history.append(count_rate)
    event_result = ed.add_measurement(
        measurement_id=measurement_id,
        measurement=gross_spectrum,
        duration=SAMPLE_INTERVAL,
    )
    if event_result:
        break


if event_result:
    event_measurement, event_bg_measurement, event_duration, event_measurement_ids = event_result
    print(f"Event Duration: {event_duration}")
    event_begin, event_end = event_measurement_ids[0], event_measurement_ids[-1]
    fix, ax = plt.subplots()
    count_rates = passby_ss.spectra.sum(axis=1) / SAMPLE_INTERVAL
    OFFSET = int(60 / SAMPLE_INTERVAL)
    times = np.array(list(range(measurement_id - OFFSET, measurement_id))) * SAMPLE_INTERVAL
    ax.plot(times, count_rate_history[-OFFSET:])
    ax.axvspan(
        xmin=event_begin * SAMPLE_INTERVAL,
        xmax=event_end * SAMPLE_INTERVAL,
        facecolor=cm.tab10(1),
        alpha=0.5,
    )
    ax.set_ylabel("Count rate (cps)")
    ax.set_xlabel("Time")
    plt.show()
else:
    print("Pass-by did not produce an event")
