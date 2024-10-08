# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the anomaly module."""
import unittest

import numpy as np

from riid import PassbySynthesizer, SeedMixer, get_dummy_seeds
from riid.anomaly import PoissonNChannelEventDetector


class TestAnomaly(unittest.TestCase):
    """Test class for Anomaly module."""
    def setUp(self):
        """Test setup."""
        pass

    def test_event_detector(self):
        random_state = 42
        rng = np.random.default_rng(random_state)

        SAMPLE_INTERVAL = 0.5
        BG_RATE = 300
        seeds_ss = get_dummy_seeds(100)
        fg_seeds_ss, bg_seeds_ss = seeds_ss.split_fg_and_bg()
        mixed_bg_seeds_ss = SeedMixer(bg_seeds_ss, mixture_size=3, rng=rng)\
            .generate(1)
        events = PassbySynthesizer(events_per_seed=1,
                                   sample_interval=SAMPLE_INTERVAL,
                                   bg_cps=BG_RATE,
                                   fwhm_function_args=(5,),
                                   dwell_time_function_args=(20, 20),
                                   snr_function_args=(20, 20),
                                   return_gross=True,
                                   rng=rng)\
            .generate(fg_seeds_ss, mixed_bg_seeds_ss, verbose=False)

        _, gross_events = list(zip(*events))
        passby_ss = gross_events[0]

        expected_bg_counts = SAMPLE_INTERVAL * BG_RATE
        expected_bg_measurement = mixed_bg_seeds_ss.spectra.iloc[0] * expected_bg_counts
        ed = PoissonNChannelEventDetector(
            long_term_duration=600,
            short_term_duration=10,
            pre_event_duration=1,
            max_event_duration=120,
            post_event_duration=10,
            tolerable_false_alarms_per_day=1e-5,
            anomaly_threshold_update_interval=60,
        )
        cps_history = []

        # Filling background
        measurement_id = 0
        while ed.background_percent_complete < 100:
            noisy_bg_measurement = np.random.poisson(expected_bg_measurement)
            cps_history.append(noisy_bg_measurement.sum() / SAMPLE_INTERVAL)
            _ = ed.add_measurement(
                measurement_id,
                noisy_bg_measurement,
                SAMPLE_INTERVAL,
                verbose=False
            )
            measurement_id += 1

        # Create event using a synthesized passby
        for i in range(passby_ss.n_samples):
            gross_spectrum = passby_ss.spectra.iloc[i].values
            cps_history.append(gross_spectrum.sum() / SAMPLE_INTERVAL)
            event_result = ed.add_measurement(
                measurement_id=measurement_id,
                measurement=gross_spectrum,
                duration=SAMPLE_INTERVAL,
                verbose=False
            )
            measurement_id += 1
            if event_result:
                break

        self.assertTrue(event_result is not None)
