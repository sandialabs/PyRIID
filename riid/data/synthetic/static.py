# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This modules contains utilities for synthesizing gamma spectra as static collections."""
import random
from collections import Counter
from datetime import datetime
from time import time
from typing import Any, Tuple

import numpy as np
import pandas as pd
from riid.data import SampleSet
from riid.data.labeling import BACKGROUND_LABEL
from riid.data.sampleset import SpectraState


class StaticSynthesizer():
    """Creates a set of synthetic gamma spectra."""
    _supported_functions = ["uniform", "log10", "discrete", "list"]

    def __init__(self, samples_per_seed: int = 100, background_cps: float = 300.0,
                 live_time_function: str = "uniform", live_time_function_args=(0.25, 8.0),
                 snr_function: str = "uniform", snr_function_args=(0.01, 100.0),
                 apply_poisson_noise: bool = True, balance_level: bool = "Seed",
                 random_state: int = None) -> None:
        """Constructs a synthetic gamma spectra generator.

        Arguments:
            samples_per_seed: Defines the number of synthetic samples to randomly generate
                per seed.
            background_cps: Defines the constant rate of gammas from background.
            live_time_function: Defines the string that names the method of sampling
                for target live time values. Options: uniform; log10; discrete; list.
            live_time_function_args: Defines the range of values which are sampled in the
                fashion specified by the `live_time_function` argument.
            snr_function: Defines the string that names the method of sampling for target
                signal-to-noise ratio values. Options: uniform; log10; discrete; list.
            snr_function_args: Defines the range of values which are sampled in the fashion
                specified by the `snr_function` argument.
            apply_poisson_noise: Defines whether to apply poisson noise to the expected spectra.
            random_state: Defines the random seed value used to reproduce specific data sets.

        """
        self.samples_per_seed = samples_per_seed
        self.background_cps = background_cps
        self.live_time_function = live_time_function
        self.live_time_function_args = live_time_function_args
        self.snr_function = snr_function
        self.snr_function_args = snr_function_args
        self.apply_poisson_noise = apply_poisson_noise
        self.random_state = random_state
        self._synthesis_start_dt = None
        self._n_samples_synthesized = 0

    # region Properties

    @property
    def detector(self) -> str:
        """Get or set the unique name of the physical detector associated with the
        provided seeds. Used to determine where in the data directory to auto-cache
        the sample set.
        """
        return self._detector

    @detector.setter
    def detector(self, value: str):
        self._detector = value

    @property
    def samples_per_seed(self):
        """Get or set the number of samples to create per seed (excluding the background seed).

        Raises:
            TypeError: Raised if provided samples_per_seed is not an integer.
        """
        return self._samples_per_seed

    @samples_per_seed.setter
    def samples_per_seed(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Property 'samples_per_seed' key must be of type 'int'!")

        self._samples_per_seed = value

    @property
    def background_cps(self) -> float:
        """Get or set the counts per second contributed by background radiation."""
        return self._background_cps

    @background_cps.setter
    def background_cps(self, value: float):
        self._background_cps = value

    @property
    def background_n_cps(self) -> float:
        """Get or set the neutron rate from background."""
        return self._background_n_cps

    @background_n_cps.setter
    def background_n_cps(self, value: float):
        self._background_n_cps = value

    @property
    def subtract_background(self) -> bool:
        """Get or set the flag for whether or not to include counts from
        background in the final spectra.
        """
        return self._subtract_background

    @subtract_background.setter
    def subtract_background(self, value: bool):
        self._subtract_background = value

    @property
    def live_time_function(self) -> str:
        """Get or set the function used to randomly sample the desired live time space.

        Raises:
            ValueError: Raised when an unsupported function type is provided.
        """
        return self._live_time_function

    @live_time_function.setter
    def live_time_function(self, value: str):
        if value not in self._supported_functions:
            raise ValueError("{} is not a valid function.".format(value))
        self._live_time_function = value

    @property
    def live_time_function_args(self) -> tuple:
        """Get or set the live time space to be randomly sampled."""
        return self._live_time_function_args

    @live_time_function_args.setter
    def live_time_function_args(self, value):
        self._live_time_function_args = value

    @property
    def snr_function(self) -> str:
        """Get or set the function used to randomly sample the desired signal-to-noise (SNR)
        ratio space.

        Raises:
            ValueError: Raised when an unsupported function type is provided.
        """
        return self._snr_function

    @snr_function.setter
    def snr_function(self, value: str):
        if value not in self._supported_functions:
            raise ValueError("{} is not a valid function.".format(value))
        self._snr_function = value

    @property
    def snr_function_args(self) -> tuple:
        """Get or set the signal-to-noise (SNR) space to be randomly sampled."""
        return self._snr_function_args

    @snr_function_args.setter
    def snr_function_args(self, value):
        self._snr_function_args = value

    @property
    def random_state(self) -> int:
        """Get or set the seed for the random number generator. Used when trying to make
        reproducible SampleSets.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, value: int):
        self._random_state = value

    # endregion

    def __str__(self):
        output = "SyntheticGenerationConfig()\n"
        for k, v in sorted(vars(self).items()):
            output += "  {}: {}\n".format(k, str(v))
        return output

    def _reset_progress(self):
        self._n_samples_synthesized = 0

    def _update_and_report_progress(self, n_samples_new, n_samples_expected, batch_name):
        self._n_samples_synthesized += n_samples_new
        percent_complete = 100 * self._n_samples_synthesized / n_samples_expected
        msg = (
            f"Synthesizing ... {percent_complete:.0f}% "
            f"(currently on {batch_name})"
        )
        print("\033[K" + msg, end="\r")

    def _verify_n_samples_synthesized_equal_expected(self, actual: int, expected: int):
        assert expected == actual, (
            f"{actual} generated, but {expected} were expected. "
            "Be sure to remove any columns from your seeds' sources DataFrame that "
            "contain all zeroes.")

    def _get_sampleset(self, spectra, sources, ecal, lt_targets, snr_targets,
                       ss_spectra_type: str,
                       fg_counts, bg_counts,
                       fg_counts_expected, bg_counts_expected):
        n_samples = spectra.shape[0]
        ss = SampleSet()
        ss.spectra_state = SpectraState.Counts
        ss.spectra = pd.DataFrame(spectra)
        ss.info.description = np.full(n_samples, "")
        ss.info.timestamp = self._synthesis_start_dt
        ss.info.live_time = lt_targets
        ss.info.real_time = lt_targets
        ss.info.bg_counts_expected = bg_counts_expected
        ss.info.fg_counts_expected = fg_counts_expected
        ss.info.gross_counts_expected = bg_counts_expected + fg_counts_expected
        ss.info.snr_expected = snr_targets
        ss.info.sigma_expected = fg_counts_expected / np.sqrt(bg_counts_expected)
        ss.info.ecal_order_0 = ecal[0]
        ss.info.ecal_order_1 = ecal[1]
        ss.info.ecal_order_2 = ecal[2]
        ss.info.ecal_order_3 = ecal[3]
        ss.info.ecal_low_e = ecal[4]
        ss.info.occupancy_flag = 0
        ss.info.tag = " "  # TODO: test if this can be empty string

        if ss_spectra_type == "bg":
            ss.info.bg_counts = spectra.sum(axis=1)
            ss.info.fg_counts = 0
            ss.info.snr = 0
            ss.info.sigma = 0
        elif ss_spectra_type == "fg":
            ss.info.bg_counts = 0
            ss.info.fg_counts = spectra.sum(axis=1)
            ss.info.snr = ss.info.fg_counts
            ss.info.sigma = ss.info.fg_counts
        elif ss_spectra_type == "gross":
            ss.info.bg_counts = bg_counts
            ss.info.fg_counts = fg_counts
            ss.info.snr = fg_counts / bg_counts
            ss.info.sigma = fg_counts / np.sqrt(bg_counts)
        else:
            raise ValueError("A non-count-based spectra state is permitted in synthesis.")

        ss.info.gross_counts = ss.info.bg_counts + ss.info.fg_counts

        if ss_spectra_type == "gross":
            ss.sources = sources
        else:
            ss.sources = pd.DataFrame(
                np.tile(sources.values, (n_samples, 1)),
                columns=sources.index
            )
            # Multiplying normalized seed source values by spectrum counts.
            # This is REQUIRED for properly merging sources DataFrames later and multi-isotope.
            ss.sources = ss.sources.multiply(spectra.sum(axis=1), axis="index")

        return ss

    def _get_fg_and_or_bg_batch(self, fg_seed, fg_sources, bg_seed, bg_sources, ecal,
                                lt_targets, snr_targets):
        bg_counts_expected = lt_targets * self.background_cps
        fg_counts_expected = snr_targets * bg_counts_expected

        fg_spectra_expected = get_expected_spectra(fg_seed.values, fg_counts_expected)
        bg_spectra_expected = get_expected_spectra(bg_seed.values, bg_counts_expected)
        if self.apply_poisson_noise:
            gross_spectra = np.random.poisson(fg_spectra_expected + bg_spectra_expected)
            bg_spectra = np.random.poisson(bg_spectra_expected)
            fg_spectra = gross_spectra - bg_spectra
        else:
            fg_spectra = fg_spectra_expected
            bg_spectra = bg_spectra_expected
            gross_spectra = fg_spectra_expected + bg_spectra_expected
        fg_ss = self._get_sampleset(fg_spectra, fg_sources, ecal,
                                    lt_targets, snr_targets, "fg",
                                    None, None, fg_counts_expected, bg_counts_expected)
        bg_ss = self._get_sampleset(bg_spectra, bg_sources, ecal,
                                    lt_targets, snr_targets, "bg",
                                    None, None, fg_counts_expected, bg_counts_expected)
        gross_sources = get_merged_sources_samplewise(fg_ss.sources, bg_ss.sources)
        gross_ss = self._get_sampleset(gross_spectra, gross_sources, ecal,
                                       lt_targets, snr_targets, "gross",
                                       fg_spectra.sum(axis=1), bg_spectra.sum(axis=1),
                                       fg_counts_expected, bg_counts_expected)

        return fg_ss, bg_ss, gross_ss

    def _get_synthetic_samples(self, fg_seeds_ss: SampleSet, bg_seeds_ss: SampleSet):
        n_samples_expected = self.samples_per_seed * fg_seeds_ss.n_samples * bg_seeds_ss.n_samples
        # Iterate over each background then each unique level value, generating a batch for each
        # seed within that level
        fg_ss_batches = []
        bg_ss_batches = []
        gross_ss_batches = []

        fg_labels = fg_seeds_ss.get_labels(target_level="Seed", max_only=False,
                                           level_aggregation=None)
        for b in range(bg_seeds_ss.n_samples):
            lt_targets = get_distribution_values(self.live_time_function,
                                                 self.live_time_function_args,
                                                 self.samples_per_seed)
            snr_targets = get_distribution_values(self.snr_function,
                                                  self.snr_function_args,
                                                  self.samples_per_seed)
            bg_seed = bg_seeds_ss.spectra.iloc[b]
            bg_sources = bg_seeds_ss.sources.iloc[b]
            for f in range(fg_seeds_ss.n_samples):
                fg_seed = fg_seeds_ss.spectra.iloc[f]
                fg_sources = fg_seeds_ss.sources.iloc[f]
                ecal = fg_seeds_ss.ecal[f]
                fg_batch_ss, bg_batch_ss, gross_batch_ss = self._get_fg_and_or_bg_batch(
                    fg_seed, fg_sources,
                    bg_seed, bg_sources,
                    ecal, lt_targets, snr_targets
                )
                fg_ss_batches.append(fg_batch_ss)
                if bg_batch_ss:
                    bg_ss_batches.append(bg_batch_ss)
                if gross_batch_ss:
                    gross_ss_batches.append(gross_batch_ss)
                self._update_and_report_progress(fg_batch_ss.n_samples, n_samples_expected,
                                                 fg_labels[f])

        fg_ss = SampleSet()
        fg_ss.measured_or_synthetic = "synthetic"
        fg_ss.concat(fg_ss_batches)
        bg_ss = SampleSet()
        bg_ss.measured_or_synthetic = "synthetic"
        bg_ss.concat(bg_ss_batches)
        gross_ss = SampleSet()
        gross_ss.measured_or_synthetic = "synthetic"
        gross_ss.concat(gross_ss_batches)

        self._verify_n_samples_synthesized_equal_expected(gross_ss.n_samples, n_samples_expected)

        return fg_ss, bg_ss, gross_ss

    def generate(self, fg_seeds_ss: SampleSet, bg_seeds_ss: SampleSet,
                 normalize_sources=True) -> Tuple[SampleSet, SampleSet, SampleSet]:
        """Generate a sample set of gamma spectra from the given config.

        Args:
            fg_seeds_ss: Contains spectra normalized by total counts to be used
                as the foreground (source only) component of spectra.
            bg_seeds_ss: Contains spectra normalized by total counts to be used
                as the background component of gross spectra.
            normalize_sources: Whether to divide each row of the SampleSet's sources
                DataFrame by its sum. Defaults to True.

        Returns:
            A tuple of synthetic foreground, background, and gross spectra.

        Raises:
            ValueError: Raised when a seed spectrum does not sum close to 1.

        """
        if not fg_seeds_ss or not bg_seeds_ss:
            raise ValueError("At least one foreground and background seed must be provided.")
        if not fg_seeds_ss.all_spectra_sum_to_one():
            raise ValueError("At least one provided foreground seed does not sum close to 1.")
        if not bg_seeds_ss.all_spectra_sum_to_one():
            raise ValueError("At least one provided background seed does not sum close to 1.")

        if self.random_state:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        self._reset_progress()
        self._synthesis_start_dt = datetime.utcnow().isoformat(sep=' ', timespec="seconds")
        tstart = time()

        fg_ss, bg_ss, gross_ss = self._get_synthetic_samples(fg_seeds_ss, bg_seeds_ss)

        if normalize_sources:
            fg_ss.normalize_sources()
            bg_ss.normalize_sources()
            gross_ss.normalize_sources()

        delay = time() - tstart
        summary = (
            f"Synthesis complete!\n"
            f"Generated {gross_ss.n_samples} samples in {delay:.2f}s "
            f"(~{(gross_ss.n_samples / delay):.2f} samples/sec)."
        )
        print("\033[K" + summary)

        return fg_ss, bg_ss, gross_ss


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


def get_distribution_values(function: str, function_args: Any, n_values: int):
    """Gets the values for the synthetic data distribution based
    on the sampling type used.

    Args:
        function: Defines the name of the distribution function.
        function_args: Defines the argument or collection of arguments to be
            passed to the function, if any.
        n_values: Defines the size of the distribution.

    Returns:
        The value or collection of values defining the distribution.

    Raises:
        ValueError: Raised when an unsupported function type is provided.
    """
    value = None
    if function == "uniform":
        value = np.random.uniform(*function_args, size=n_values)
    elif function == "log10":
        log10_args = tuple(map(np.log10, function_args))
        value = np.power(10, np.random.uniform(*log10_args, size=n_values))
    elif function == "discrete":
        value = np.random.choice(function_args, size=n_values)
    elif function == "list":
        value = function_args

    return value


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


def get_dummy_sampleset(n_channels: int = 512, as_seeds: bool = False,
                        live_time: int = 60, fg_rate: int = 300,
                        bg_rate: int = 300) -> SampleSet:
    """Builds a random, dummy SampleSet for demonstration or test purposes.

    Args:
        n_channels: the number of channels in the spectra DataFrame.
        as_seeds: whether or not the SampleSet should be a seed file.
            When True, the samples will be all be pure foregrounds
            and one pure background.  When False, the samples will all
            be gross measurements.
        live_time: the collection time for all measurements.
        fg_rate: the count rate for foreground measurements.
        bg_rate: the count rate for background measurements.

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
        ("NORM",        "K40",      "Unshielded K40"),
        ("NORM",        "K40",      "Moderately Shielded K40"),
        ("NORM",        "Ra226",    "Unshielded Ra226"),
        ("NORM",        "Th232",    "Unshielded Th232"),
        ("NORM",        "U238",     "Unshielded U238"),
        ("SNM",         "Pu239",    "Unshielded Pu239"),
        ("SNM",         "Pu239",    "Moderately Shielded Pu239"),
        ("SNM",         "Pu239",    "Heavily Shielded Pu239"),
    ]
    n_sources = len(sources)
    n_fg_sources = n_sources
    if as_seeds:
        sources.append((BACKGROUND_LABEL, BACKGROUND_LABEL, "k40=8%,U=8ppm,Th232=8ppm+cosmic"))
        n_sources += 1
    sources_index = pd.MultiIndex.from_tuples(
        sources,
        names=SampleSet.SOURCES_MULTI_INDEX_NAMES
    )
    sources_data = np.identity(n_sources)
    ss.sources = pd.DataFrame(data=sources_data, columns=sources_index)

    histograms = []
    N_BG_COUNTS = bg_rate * live_time
    bg_counts = np.random.uniform(0, n_channels, size=N_BG_COUNTS)
    bg_histogram, _ = np.histogram(bg_counts, bins=n_channels, range=(0, n_channels))

    N_FG_COUNTS = fg_rate * live_time
    fg_std = np.sqrt(n_channels / n_sources)
    channels_per_sources = n_channels / n_fg_sources
    for i in range(n_fg_sources):
        mu = i * channels_per_sources + channels_per_sources / 2
        counts = np.random.normal(mu, fg_std, size=N_FG_COUNTS)
        fg_histogram, _ = np.histogram(counts, bins=n_channels, range=(0, n_channels))
        histogram = fg_histogram
        if not as_seeds:
            histogram += bg_histogram
        histograms.append(histogram)
    if as_seeds:
        histograms.append(bg_histogram)
    histograms = np.array(histograms)

    ss.spectra = pd.DataFrame(data=histograms)

    ss.info.bg_counts = N_BG_COUNTS
    ss.info.bg_counts_expected = N_BG_COUNTS
    ss.info.fg_counts = N_FG_COUNTS
    ss.info.fg_counts_expected = N_FG_COUNTS
    ss.info.gross_counts = ss.spectra.sum(axis=1)
    ss.info.gross_counts_expected = ss.info.gross_counts
    ss.info.live_time = live_time
    ss.info.real_time = live_time
    ss.info.snr = N_FG_COUNTS / N_BG_COUNTS
    ss.info.sigma = N_FG_COUNTS / np.sqrt(N_BG_COUNTS)
    ss.info.ecal_order_0 = 0
    ss.info.ecal_order_1 = 3000
    ss.info.ecal_order_2 = 100
    ss.info.ecal_order_3 = 0
    ss.info.ecal_low_e = 0
    ss.info.description = ""
    ss.update_timestamp()

    if as_seeds:
        ss.to_pmf()

    return ss


class NoSeedError(Exception):
    pass


class InvalidSeedError(Exception):
    pass
