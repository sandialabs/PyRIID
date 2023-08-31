# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This modules contains utilities for synthesizing gamma spectra."""
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from numpy.random import Generator

from riid.data import get_expected_spectra
from riid.data.sampleset import (SampleSet, SpectraState, SpectraType,
                                 _get_utc_timestamp)


class Synthesizer():
    """Base class for synthesizers."""

    SYNTHETIC_STR = "synthetic"
    SUPPORTED_SAMPLING_FUNCTIONS = ["uniform", "log10", "discrete", "list"]

    def __init__(self, bg_cps: float = 300.0, long_bg_live_time: float = 120.0,
                 apply_poisson_noise: bool = True,
                 normalize_sources: bool = True,
                 return_fg: bool = True,
                 return_gross: bool = False,
                 rng: Generator = np.random.default_rng()):
        """
        Args:
            bg_cps: constant rate of gammas from background
            long_bg_live_time: live time on which to base background subtractions
            apply_poisson_noise: whether to apply Poisson noise to spectra
            normalize_sources: whether to normalize ground truth proportions to sum to 1
            return_fg: whether to compute and return background subtracted spectra
            return_gross: whether to return gross spectra (always computed)
            rng: NumPy random number generator, useful for experiment repeatability
        """
        self.bg_cps = bg_cps
        self.long_bg_live_time = long_bg_live_time
        self.apply_poisson_noise = apply_poisson_noise
        self.normalize_sources = normalize_sources
        self.return_fg = return_fg
        self.return_gross = return_gross
        self._rng = rng
        self._synthesis_start_dt = None
        self._n_samples_synthesized = 0

    def __str__(self):
        output = "SynthesizerConfig"
        for k, v in sorted(vars(self).items()):
            output += "  {}: {}".format(k, str(v))
        return output

    def _reset_progress(self):
        self._n_samples_synthesized = 0
        self._synthesis_start_dt = _get_utc_timestamp()

    def _report_progress(self, n_samples_expected, batch_name):
        percent_complete = 100 * self._n_samples_synthesized / n_samples_expected
        msg = (
            f"Synthesizing ... {percent_complete:.0f}% "
            f"(currently on {batch_name})"
        )
        print("\033[K" + msg, end="\r")

    def _report_completion(self, delay):
        summary = (
            f"Synthesis complete!\n"
            f"Generated {self._n_samples_synthesized} samples in ~{delay:.2f}s "
            f"(~{(self._n_samples_synthesized / delay):.2f} samples/sec)."
        )
        print("\033[K" + summary)

    def _verify_n_samples_synthesized(self, actual: int, expected: int):
        assert expected == actual, (
            f"{actual} generated, but {expected} were expected. "
            "Be sure to remove any columns from your seeds' sources DataFrame that "
            "contain all zeroes.")

    def _get_batch(self, fg_seed, fg_sources, bg_seed, bg_sources, ecal,
                   lt_targets, snr_targets):
        if not (self.return_fg or self.return_gross):
            raise ValueError("Computing to return nothing.")

        bg_counts_expected = lt_targets * self.bg_cps
        fg_counts_expected = snr_targets * np.sqrt(bg_counts_expected)

        fg_spectra = get_expected_spectra(fg_seed.values, fg_counts_expected)
        bg_spectra = get_expected_spectra(bg_seed.values, bg_counts_expected)

        long_bg_counts_expected = self.long_bg_live_time * self.bg_cps
        long_bg_spectrum_expected = bg_seed.values * long_bg_counts_expected

        gross_spectra = None
        long_bg_spectra = None
        fg_counts = 0
        bg_counts = 0
        long_bg_counts = 0
        fg_ss = None
        gross_ss = None

        # Spectra
        if self.apply_poisson_noise:
            gross_spectra = self._rng.poisson(fg_spectra + bg_spectra)
            if self.return_fg:
                long_bg_spectrum = self._rng.poisson(long_bg_spectrum_expected)
                long_bg_seed = long_bg_spectrum / long_bg_spectrum.sum()
                long_bg_spectra = get_expected_spectra(long_bg_seed, bg_counts_expected)
                fg_spectra = gross_spectra - long_bg_spectra
        else:
            gross_spectra = fg_spectra + bg_spectra
            if self.return_fg:
                long_bg_spectra = bg_spectra
                fg_spectra = gross_spectra - long_bg_spectra

        # Counts
        fg_counts = fg_spectra.sum(axis=1, dtype=float)
        if self.return_fg:
            long_bg_counts = long_bg_spectra.sum(axis=1, dtype=float)
        if self.return_gross:
            bg_counts = bg_spectra.sum(axis=1, dtype=float)

        # Sample sets
        if self.return_fg:
            snrs = fg_counts / np.sqrt(long_bg_counts.clip(1))
            fg_ss = get_fg_sample_set(fg_spectra, fg_sources, ecal, lt_targets,
                                      snrs=snrs, total_counts=fg_counts,
                                      timestamps=self._synthesis_start_dt)
            self._n_samples_synthesized += fg_ss.n_samples
        if self.return_gross:
            tiled_fg_sources = _tile_sources_and_scale(
                fg_sources,
                gross_spectra.shape[0],
                fg_counts,
            )
            tiled_bg_sources = _tile_sources_and_scale(
                bg_sources,
                gross_spectra.shape[0],
                bg_counts,
            )
            gross_sources = get_merged_sources_samplewise(tiled_fg_sources, tiled_bg_sources)
            gross_counts = gross_spectra.sum(axis=1)
            snrs = fg_counts / np.sqrt(bg_counts.clip(1))
            gross_ss = get_gross_sample_set(gross_spectra, gross_sources, ecal,
                                            lt_targets, snrs, gross_counts,
                                            timestamps=self._synthesis_start_dt)
            self._n_samples_synthesized += gross_ss.n_samples

        return fg_ss, gross_ss


def get_sample_set(spectra, sources, ecal, live_times, snrs, total_counts=None,
                   real_times=None, timestamps=None, descriptions=None) -> SampleSet:
    n_samples = spectra.shape[0]

    ss = SampleSet()
    ss.spectra_state = SpectraState.Counts
    ss.spectra = pd.DataFrame(spectra)
    ss.sources = sources
    ss.info.description = np.full(n_samples, "")  # Ensures the length of info equal n_samples
    if descriptions:
        ss.info.description = descriptions
    ss.info.snr = snrs
    ss.info.timestamp = timestamps
    ss.info.total_counts = total_counts if total_counts is not None else spectra.sum(axis=1)
    ss.info.ecal_order_0 = ecal[0]
    ss.info.ecal_order_1 = ecal[1]
    ss.info.ecal_order_2 = ecal[2]
    ss.info.ecal_order_3 = ecal[3]
    ss.info.ecal_low_e = ecal[4]
    ss.info.live_time = live_times
    ss.info.real_time = real_times if real_times is not None else live_times
    ss.info.occupancy_flag = 0
    ss.info.tag = " "  # TODO: test if this can be empty string

    return ss


def _tile_sources_and_scale(sources, n_samples, scalars) -> pd.DataFrame:
    tiled_sources = pd.DataFrame(
        np.tile(sources.values, (n_samples, 1)),
        columns=sources.index
    )
    # Multiplying normalized source values by spectrum counts.
    # This is REQUIRED for properly merging sources DataFrames later when synthesizing
    # multiple isotopes.
    tiled_sources = tiled_sources.multiply(scalars, axis="index")
    return tiled_sources


def get_fg_sample_set(spectra, sources, ecal, live_times, snrs, total_counts,
                      real_times=None, timestamps=None, descriptions=None) -> SampleSet:
    tiled_sources = _tile_sources_and_scale(
        sources,
        spectra.shape[0],
        spectra.sum(axis=1)
    )
    ss = get_sample_set(
        spectra=spectra,
        sources=tiled_sources,
        ecal=ecal,
        live_times=live_times,
        snrs=snrs,
        total_counts=total_counts,
        real_times=real_times,
        timestamps=timestamps,
        descriptions=descriptions
    )
    ss.spectra_type = SpectraType.Foreground
    return ss


def get_gross_sample_set(spectra, sources, ecal, live_times, snrs, total_counts,
                         real_times=None, timestamps=None, descriptions=None) -> SampleSet:
    ss = get_sample_set(
        spectra=spectra,
        sources=sources,
        ecal=ecal,
        live_times=live_times,
        snrs=snrs,
        total_counts=total_counts,
        real_times=real_times,
        timestamps=timestamps,
        descriptions=descriptions
    )
    ss.spectra_type = SpectraType.Gross
    return ss


def get_distribution_values(function: str, function_args: Any, n_values: int,
                            rng: Generator = np.random.default_rng()):
    """Randomly sample a list of values based one of many distributions.

    Args:
        function: name of the distribution function
        function_args: argument or collection of arguments to be
            passed to the function, if any.
        n_values: size of the distribution
        rng: NumPy random number generator, useful for experiment repeatability

    Returns:
        Value or collection of sampled values

    Raises:
        `ValueError` when an unsupported function type is provided
    """
    values = None
    if function == "uniform":
        values = rng.uniform(*function_args, size=n_values)
    elif function == "log10":
        log10_args = tuple(map(np.log10, function_args))
        values = np.power(10, rng.uniform(*log10_args, size=n_values))
    elif function == "discrete":
        values = rng.choice(function_args, size=n_values)
    elif function == "list":
        values = function_args
    else:
        raise ValueError(f"{function} function not supported for sampling.")

    return values


def get_merged_sources_samplewise(sources1: pd.DataFrame, sources2: pd.DataFrame) -> pd.DataFrame:
    merged_sources_df = sources1.add(sources2, axis=1, fill_value=0)
    return merged_sources_df


def get_samples_per_seed(columns: pd.MultiIndex, min_samples_per_seed: int, balance_level: int):
    level_values = columns.get_level_values(level=balance_level)
    level_value_to_n_seeds = Counter(level_values)
    unique_level_values = list(level_value_to_n_seeds.keys())
    occurences = np.array(list(level_value_to_n_seeds.values()))
    max_samples_per_level_value = occurences.max() * min_samples_per_seed
    samples_per_level_value = np.ceil(max_samples_per_level_value / occurences).astype(int)
    lv_to_samples_per_seed = {k: v for (k, v) in zip(unique_level_values, samples_per_level_value)}
    total_samples_expected = sum([x * y for x, y in zip(occurences, samples_per_level_value)])

    return lv_to_samples_per_seed, total_samples_expected


def get_dummy_seeds(n_channels: int = 512, live_time: float = 600.0,
                    count_rate: float = 1000.0, normalize: bool = True,
                    rng: Generator = np.random.default_rng()) -> SampleSet:
    """Get a random, dummy `SampleSet` for demonstration or test purposes.

    Args:
        n_channels: number of channels in the spectra DataFrame
        live_time: collection time on which to base seeds
            (higher creates a less noisy shape)
        count_rate: count rate on which to base seeds
            (higher creates a less noisy shape)
        normalize: whether to apply an L1-norm to the spectra
        rng: NumPy random number generator, useful for experiment repeatability

    Returns:
        `SampleSet` with randomly generated spectra
    """
    ss = SampleSet()
    ss.measured_or_synthetic = "synthetic"
    ss.spectra_state = SpectraState.Counts
    ss.spectra_type = SpectraType.BackgroundForeground
    ss.synthesis_info = {
        "subtract_background": True,
    }
    sources = [
        ("Industrial",  "Am241",    "Unshielded Am241"),
        ("Industrial",  "Ba133",    "Unshielded Ba133"),
        ("NORM",        "K40",      "PotassiumInSoil"),
        ("NORM",        "K40",      "Moderately Shielded K40"),
        ("NORM",        "Ra226",    "UraniumInSoil"),
        ("NORM",        "Th232",    "ThoriumInSoil"),
        ("SNM",         "U238",     "Unshielded U238"),
        ("SNM",         "Pu239",    "Unshielded Pu239"),
        ("SNM",         "Pu239",    "Moderately Shielded Pu239"),
        ("SNM",         "Pu239",    "Heavily Shielded Pu239"),
    ]
    n_sources = len(sources)
    n_fg_sources = n_sources
    sources_cols = pd.MultiIndex.from_tuples(
        sources,
        names=SampleSet.SOURCES_MULTI_INDEX_NAMES
    )
    sources_data = np.identity(n_sources)
    ss.sources = pd.DataFrame(data=sources_data, columns=sources_cols)

    histograms = []
    N_FG_COUNTS = int(count_rate * live_time)
    fg_std = np.sqrt(n_channels / n_sources)
    channels_per_sources = n_channels / n_fg_sources
    for i in range(n_fg_sources):
        mu = i * channels_per_sources + channels_per_sources / 2
        counts = rng.normal(mu, fg_std, size=N_FG_COUNTS)
        fg_histogram, _ = np.histogram(counts, bins=n_channels, range=(0, n_channels))
        histograms.append(fg_histogram)
    histograms = np.array(histograms)

    ss.spectra = pd.DataFrame(data=histograms)

    ss.info.total_counts = ss.spectra.sum(axis=1)
    ss.info.live_time = live_time
    ss.info.real_time = live_time
    ss.info.snr = None
    ss.info.ecal_order_0 = 0
    ss.info.ecal_order_1 = 3000
    ss.info.ecal_order_2 = 100
    ss.info.ecal_order_3 = 0
    ss.info.ecal_low_e = 0
    ss.info.description = ""
    ss.update_timestamp()

    if normalize:
        ss.normalize()

    return ss
