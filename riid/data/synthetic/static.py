# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains utilities for synthesizing gamma spectra
based on a source moving past a detector.
"""
from time import time
from typing import Tuple

import numpy as np
from numpy.random import Generator

from riid.data.sampleset import SampleSet, SpectraState
from riid.data.synthetic import Synthesizer, get_distribution_values


class StaticSynthesizer(Synthesizer):
    """Creates a set of synthetic gamma spectra from seed templates.

    The "seed" templates are count-normalized spectra representing signature shapes of interest.
    The static synthesizer takes the seeds you have chosen and scales them up in terms of three
    components:

    - live time (the amount of time over which the spectrum was collected/integrated)
    - signal-to-noise ratio (SNR) (source counts divided by the square root of background counts,
      i.e., the number of standard deviations above background)
    - a fixed background rate (as a reference point)

    The static synthesizer is meant to capture various count rates from sources in a statically
    placed detector scenario where the background can be characterized (and usually subtracted).
    Effects related to pile-up, scattering, or other shape-changing outcomes must be represented
    in the seeds.
    """

    def __init__(self, samples_per_seed: int = 100,
                 live_time_function: str = "uniform", live_time_function_args=(0.25, 8.0),
                 snr_function: str = "uniform", snr_function_args=(0.01, 100.0),
                 bg_cps: float = 300.0, long_bg_live_time: float = 120.0,
                 apply_poisson_noise: bool = True,
                 normalize_sources: bool = True,
                 return_fg: bool = True, return_gross: bool = False,
                 rng: Generator = np.random.default_rng()) -> None:
        """
        Args:
            samples_per_seed: number of synthetic samples to generate
                per source-background seed pair
            live_time_function: string that names the method of sampling
                for target live time values (options: uniform, log10, discrete, list)
            live_time_function_args: range of values which are sampled in the
                fashion specified by the `live_time_function` argument
            snr_function: string that names the method of sampling for target
                signal-to-noise ratio values (options: uniform, log10, discrete, list)
            snr_function_args: range of values which are sampled in the fashion
                specified by the `snr_function` argument
        """
        super().__init__(bg_cps, long_bg_live_time, apply_poisson_noise, normalize_sources,
                         return_fg, return_gross, rng)

        self.samples_per_seed = samples_per_seed
        self.live_time_function = live_time_function
        self.live_time_function_args = live_time_function_args
        self.snr_function = snr_function
        self.snr_function_args = snr_function_args

    # region Properties

    @property
    def samples_per_seed(self):
        """Get or set the number of samples to create per seed (excluding the background seed).

        Raises:
            `TypeError` when provided samples_per_seed is not an integer.
        """
        return self._samples_per_seed

    @samples_per_seed.setter
    def samples_per_seed(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Property 'samples_per_seed' key must be of type 'int'!")

        self._samples_per_seed = value

    @property
    def live_time_function(self) -> str:
        """Get or set the function used to randomly sample the desired live time space.

        Raises:
            `ValueError` when an unsupported function type is provided.
        """
        return self._live_time_function

    @live_time_function.setter
    def live_time_function(self, value: str):
        if value not in self.SUPPORTED_SAMPLING_FUNCTIONS:
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
            `ValueError` when an unsupported function type is provided.
        """
        return self._snr_function

    @snr_function.setter
    def snr_function(self, value: str):
        if value not in self.SUPPORTED_SAMPLING_FUNCTIONS:
            raise ValueError("{} is not a valid function.".format(value))
        self._snr_function = value

    @property
    def snr_function_args(self) -> tuple:
        """Get or set the signal-to-noise (SNR) space to be randomly sampled."""
        return self._snr_function_args

    @snr_function_args.setter
    def snr_function_args(self, value):
        self._snr_function_args = value

    # endregion

    def _get_concatenated_batches(self, ss_batches, n_samples_expected):
        ss = SampleSet()
        ss.measured_or_synthetic = self.SYNTHETIC_STR
        ss.spectra_state = SpectraState.Counts
        ss.concat(ss_batches)
        self._verify_n_samples_synthesized(ss.n_samples, n_samples_expected)
        if self.normalize_sources:
            ss.normalize_sources()
        return ss

    def _get_synthetic_samples(self, fg_seeds_ss: SampleSet, bg_seeds_ss: SampleSet, verbose=True):
        """Iterate over each background, then each source, to generate a batch of spectra that
            target a set of SNR and live time values.
        """
        n_returns = self.return_fg + self.return_gross
        n_samples_per_return = self.samples_per_seed * fg_seeds_ss.n_samples * bg_seeds_ss.n_samples
        n_samples_expected = n_returns * n_samples_per_return
        fg_ss_batches = []
        gross_ss_batches = []

        fg_labels = fg_seeds_ss.get_labels(target_level="Seed", max_only=False,
                                           level_aggregation=None)
        for b in range(bg_seeds_ss.n_samples):
            lt_targets = get_distribution_values(self.live_time_function,
                                                 self.live_time_function_args,
                                                 self.samples_per_seed,
                                                 self._rng)
            snr_targets = get_distribution_values(self.snr_function,
                                                  self.snr_function_args,
                                                  self.samples_per_seed,
                                                  self._rng)
            bg_seed = bg_seeds_ss.spectra.iloc[b]
            bg_sources = bg_seeds_ss.sources.iloc[b]
            for f in range(fg_seeds_ss.n_samples):
                fg_seed = fg_seeds_ss.spectra.iloc[f]
                fg_sources = fg_seeds_ss.sources.iloc[f]
                ecal = fg_seeds_ss.ecal[f]
                fg_batch_ss, gross_batch_ss = self._get_batch(
                    fg_seed, fg_sources,
                    bg_seed, bg_sources,
                    ecal, lt_targets, snr_targets
                )
                fg_ss_batches.append(fg_batch_ss)
                gross_ss_batches.append(gross_batch_ss)

                if verbose:
                    self._report_progress(
                        n_samples_expected,
                        fg_labels[f]
                    )

        fg_ss = gross_ss = None
        if self.return_fg:
            fg_ss = self._get_concatenated_batches(fg_ss_batches, n_samples_per_return)
        if self.return_gross:
            gross_ss = self._get_concatenated_batches(gross_ss_batches, n_samples_per_return)

        return fg_ss, gross_ss

    def generate(self, fg_seeds_ss: SampleSet, bg_seeds_ss: SampleSet,
                 verbose: bool = True) -> Tuple[SampleSet, SampleSet, SampleSet]:
        """Generate a `SampleSet` of gamma spectra from the provided config.

        Args:
            fg_seeds_ss: spectra normalized by total counts to be used
                as the source component(s) of spectra
            bg_seeds_ss: spectra normalized by total counts to be used
                as the background components of gross spectra
            fixed_bg_ss: single spectrum to be used as a fixed (or intrinsic)
                background source; live time information must be present.
                This spectrum is used to represent things like:

                - cosmic background (which is location-specific);
                - one or more calibration sources; or
                - intrinsic counts from the detector material (e.g., LaBr3).

                This spectrum will form the base of each background spectrum where seeds in
                `bg_seeds_ss`, which represent mixtures of K-U-T, get added on top.
                Note: this spectrum is not considered part of the `bg_cps` parameter,
                but is instead added on top of it.
            verbose: whether to show detailed output

        Returns:
            Tuple of synthetic foreground, background, and gross spectra

        Raises:
            `ValueError` when a seed spectrum does not sum close to 1
        """
        if not fg_seeds_ss or not bg_seeds_ss:
            raise ValueError("At least one foreground and background seed must be provided.")
        if not fg_seeds_ss.all_spectra_sum_to_one():
            raise ValueError("At least one provided foreground seed does not sum close to 1.")
        if not bg_seeds_ss.all_spectra_sum_to_one():
            raise ValueError("At least one provided background seed does not sum close to 1.")

        self._reset_progress()
        if verbose:
            tstart = time()

        fg_ss, gross_ss = self._get_synthetic_samples(
            fg_seeds_ss,
            bg_seeds_ss,
            verbose=verbose
        )

        if verbose:
            delay = time() - tstart
            self._report_completion(delay)

        return fg_ss, gross_ss


class NoSeedError(Exception):
    pass
