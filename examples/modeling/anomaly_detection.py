# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to obtain events using an anomaly detection
algorithm.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from riid.anomaly import PoissonNChannelEventDetector
from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.passby import PassbySynthesizer
from riid.data.synthetic.seed import SeedMixer

if len(sys.argv) == 2:
    import matplotlib
    matplotlib.use("Agg")

SAMPLE_INTERVAL = 0.5
BG_RATE = 300
EXPECTED_BG_COUNTS = SAMPLE_INTERVAL * BG_RATE
SHORT_TERM_DURATION = 1.5
POST_EVENT_DURATION = 1.5
N_POST_EVENT_SAMPLES = (POST_EVENT_DURATION + SAMPLE_INTERVAL) / SAMPLE_INTERVAL

fg_seeds_ss, bg_seeds_ss = get_dummy_seeds().split_fg_and_bg()
mixed_bg_seed_ss = SeedMixer(bg_seeds_ss, mixture_size=3)\
    .generate(1)

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
expected_bg_measurement = mixed_bg_seed_ss.spectra.iloc[0] * EXPECTED_BG_COUNTS
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

# Now let's make a passby
print("Generating pass-by")
events = PassbySynthesizer(
    fwhm_function_args=(1,),
    snr_function_args=(30, 30),
    dwell_time_function_args=(1, 1),
    events_per_seed=1,
    sample_interval=SAMPLE_INTERVAL,
    bg_cps=BG_RATE,
    return_fg=False,
    return_gross=True,
).generate(fg_seeds_ss, mixed_bg_seed_ss)
_, _, gross_events = list(zip(*events))
passby_ss = gross_events[0]

print("Passing by...")
passby_begin_idx = measurement_id
passby_end_idx = passby_ss.n_samples + measurement_id
passby_range = list(range(passby_begin_idx, passby_end_idx))
for i, measurement_id in enumerate(passby_range):
    gross_spectrum = passby_ss.spectra.iloc[i].values
    count_rate = gross_spectrum.sum() / SAMPLE_INTERVAL
    count_rate_history.append(count_rate)
    event_result = ed.add_measurement(
        measurement_id=measurement_id,
        measurement=gross_spectrum,
        duration=SAMPLE_INTERVAL,
    )
    if event_result:
        break

# A little extra background to close out any pending event
if ed.event_in_progress:
    while not event_result:
        measurement_id += 1
        noisy_bg_measurement = np.random.poisson(expected_bg_measurement)
        count_rate = noisy_bg_measurement.sum() / SAMPLE_INTERVAL
        count_rate_history.append(count_rate)
        event_result = ed.add_measurement(
            measurement_id,
            noisy_bg_measurement,
            SAMPLE_INTERVAL
        )

count_rate_history = np.array(count_rate_history)
if event_result:
    event_measurement, event_bg_measurement, event_duration, event_measurement_ids = event_result
    print(f"Event Duration: {event_duration}")
    event_begin, event_end = event_measurement_ids[0], event_measurement_ids[-1]
    start_idx = int(passby_begin_idx - 30 / SAMPLE_INTERVAL)  # include some lead up to event
    y = count_rate_history[start_idx:measurement_id]
    x = np.array(range(start_idx, measurement_id)) * SAMPLE_INTERVAL
    fix, ax = plt.subplots()
    ax.plot(x, y)
    ax.axvspan(
        xmin=event_begin * SAMPLE_INTERVAL,
        xmax=event_end * SAMPLE_INTERVAL,
        facecolor=cm.tab10(1),
        alpha=0.35,
    )
    ax.set_ylabel("Count rate (cps)")
    ax.set_xlabel("Time (sec)")
    plt.show()
else:
    print("Pass-by did not produce an event")
