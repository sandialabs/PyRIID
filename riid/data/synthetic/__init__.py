# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This modules contains utilities for synthesizing gamma spectra."""
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from numpy.random import Generator

from riid.data.sampleset import SampleSet, SpectraState, _get_utc_timestamp


class Synthesizer():
    """Base synthesizer."""
    SYNTHETIC_STR = "synthetic"
    SUPPORTED_SAMPLING_FUNCTIONS = ["uniform", "log10", "discrete", "list"]

    def __init__(self, bg_cps: float = 300.0,
                 apply_poisson_noise: bool = True,
                 normalize_sources: bool = True,
                 return_fg: bool = True,
                 return_bg: bool = False,
                 return_gross: bool = False,
                 rng: Generator = np.random.default_rng()):
        self.bg_cps = bg_cps
        self.apply_poisson_noise = apply_poisson_noise
        self.normalize_sources = normalize_sources
        self.return_fg = return_fg
        self.return_bg = return_bg
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
        if not (self.return_fg or self.return_bg or self.return_gross):
            raise ValueError("Computing to return nothing.")

        bg_counts_expected = lt_targets * self.bg_cps
        fg_counts_expected = snr_targets * np.sqrt(bg_counts_expected)
        fg_spectra = get_expected_spectra(fg_seed.values, fg_counts_expected)
        bg_spectra = get_expected_spectra(bg_seed.values, bg_counts_expected)
        gross_spectra = None
        fg_counts = 0
        bg_counts = 0
        fg_ss = None
        bg_ss = None
        gross_ss = None

        # Spectra
        if self.apply_poisson_noise:
            if self.return_fg or self.return_gross:
                gross_spectra = self._rng.poisson(fg_spectra + bg_spectra)
            if self.return_fg or self.return_bg:
                bg_spectra = self._rng.poisson(bg_spectra)
            if self.return_fg:
                fg_spectra = gross_spectra - bg_spectra
        elif self.return_gross:
            gross_spectra = fg_spectra + bg_spectra

        # Counts
        if self.return_fg or self.return_gross:
            fg_counts = fg_spectra.sum(axis=1, dtype=float)
        if self.return_bg or self.return_gross:
            bg_counts = bg_spectra.sum(axis=1, dtype=float)

        # Sample sets
        if self.return_fg:
            fg_ss = get_fg_sample_set(fg_spectra, fg_sources, ecal, lt_targets,
                                      snrs=fg_counts, total_counts=fg_counts,
                                      timestamps=self._synthesis_start_dt)
            self._n_samples_synthesized += fg_ss.n_samples
        if self.return_bg:
            bg_ss = get_bg_sample_set(bg_spectra, bg_sources, ecal, lt_targets,
                                      snrs=0, total_counts=bg_counts,
                                      timestamps=self._synthesis_start_dt)
            self._n_samples_synthesized += bg_ss.n_samples
        if self.return_gross:
            gross_sources = get_merged_sources_samplewise(
                _tile_sources_and_scale(
                    fg_sources,
                    gross_spectra.shape[0],
                    fg_counts,
                ),
                _tile_sources_and_scale(
                    bg_sources,
                    gross_spectra.shape[0],
                    bg_counts,
                ),
            )
            gross_counts = gross_spectra.sum(axis=1)
            snrs = fg_counts / np.sqrt(bg_counts.clip(1))
            gross_ss = get_gross_sample_set(gross_spectra, gross_sources, ecal,
                                            lt_targets, snrs, gross_counts,
                                            timestamps=self._synthesis_start_dt)
            self._n_samples_synthesized += gross_ss.n_samples

        return fg_ss, bg_ss, gross_ss


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


def get_bg_sample_set(spectra, sources, ecal, live_times, snrs, total_counts,
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
    return ss


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
    return ss


def get_distribution_values(function: str, function_args: Any, n_values: int,
                            rng: Generator = np.random.default_rng()):
    """Gets the values for the synthetic data distribution based
    on the sampling type used.

    Args:
        function: Defines the name of the distribution function.
        function_args: Defines the argument or collection of arguments to be
            passed to the function, if any.
        n_values: Defines the size of the distribution.
        rng: a NumPy random number generator, useful for experiment repeatability.

    Returns:
        The value or collection of values defining the distribution.

    Raises:
        ValueError: Raised when an unsupported function type is provided.
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


def get_expected_spectra(seeds: np.ndarray, expected_counts: np.ndarray) -> np.ndarray:
    """ Multiples a 1-D array of expected counts by either a 1-D array or 2-D
        matrix of seed spectra.

        The dimension(s) of the seed array(s), `seeds`, is expanded to be (m, n, 1) where:
            m = # of seeds
            n = # of channels
            and the final dimension is added in order to facilitate proper broadcasting
        The dimension of the `expected_counts` must be 1, but the length `p` can be
        any positive number.

        The resulting expected spectra will be of shape (m x p, n).
        This representings the same number of channels `n`, but each expected count
        value, of which there were `p`, will be me multiplied through each seed spectrum,
        of which there were `m`.
        All expected spectra matrices for each seed are then concatenated together
        (stacked), eliminating the 3rd dimension.
    """
    if expected_counts.ndim != 1:
        raise ValueError("Expected counts array must be 1-D.")
    if expected_counts.shape[0] == 0:
        raise ValueError("Expected counts array cannot be empty.")
    if seeds.ndim > 2:
        raise InvalidSeedError("Seeds array must be 1-D or 2-D.")

    expected_spectra = np.concatenate(
        seeds * expected_counts[:, np.newaxis, np.newaxis]
    )

    return expected_spectra


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


def get_dummy_seeds(n_channels: int = 512, live_time: float = 1,
                    count_rate: float = 100, normalize: bool = True,
                    rng: Generator = np.random.default_rng()) -> SampleSet:
    """Builds a random, dummy SampleSet for demonstration or test purposes.

    Args:
        n_channels: the number of channels in the spectra DataFrame.
        live_time: the collection time for all measurements.
        count_rate: the count rate for the seeds measurements.
        normalize: whether to apply an L1-norm to the spectra.
        rng: a NumPy random number generator, useful for experiment repeatability.

    Returns:
        A SampleSet with randomly generated spectra
    """
    ss = SampleSet()
    ss.measured_or_synthetic = "synthetic"
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
        histogram = rng.poisson(fg_histogram)
        histograms.append(histogram)
    histograms = np.array(histograms)

    ss.spectra = pd.DataFrame(data=histograms)

    ss.info["total_counts"] = ss.spectra.sum(axis=1)
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


class InvalidSeedError(Exception):
    pass
