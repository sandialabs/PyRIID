# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This modules contains utilities for synthesizing gamma spectra for a source moving past a
detector.
"""
from time import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator

from riid.data.sampleset import SampleSet
from riid.data.synthetic import Synthesizer, get_distribution_values


class PassbySynthesizer(Synthesizer):
    """Creates synthetic pass-by events as sequences of gamma spectra."""
    _supported_functions = ["uniform", "log10", "discrete", "list"]

    def __init__(self, events_per_seed: int = 2, sample_interval: float = 0.125,
                 dwell_time_function: str = "uniform", dwell_time_function_args=(0.25, 8.0),
                 fwhm_function: str = "discrete", fwhm_function_args=(1,),
                 snr_function: str = "uniform", snr_function_args=(1.0, 10.0),
                 min_fraction: float = 0.005, normalize_sources: bool = True,
                 bg_cps: float = 300.0, apply_poisson_noise: bool = True,
                 return_fg: bool = True, return_bg: bool = False, return_gross: bool = False,
                 rng: Generator = np.random.default_rng()):
        """Constructs a synthetic passy-by generator.

        Args:
            events_per_seed: Defines the number of pass-bys to generate
                per source-background seed pair.
            bg_cps: Defines the constant rate of gammas from background.
            live_time_function: Defines the string that names the method of sampling
                for target live time values. Options: uniform; log10; discrete; list.
            live_time_function_args: Defines the range of values which are sampled in the
                fashion specified by the `live_time_function` argument.
            snr_function: Defines the string that names the method of sampling for target
                signal-to-noise ratio values. Options: uniform; log10; discrete; list.
            snr_function_args: Defines the range of values which are sampled in the fashion
                specified by the `snr_function` argument.
            min_fraction: Minimum proportion of peak amplitude to exclude.
        """
        super().__init__(bg_cps, apply_poisson_noise, normalize_sources,
                         return_fg, return_bg, return_gross, rng)

        self.events_per_seed = events_per_seed
        self.sample_interval = sample_interval
        self.dwell_time_function = dwell_time_function
        self.dwell_time_function_args = dwell_time_function_args
        self.fwhm_function = fwhm_function
        self.fwhm_function_args = fwhm_function_args
        self.snr_function = snr_function
        self.snr_function_args = snr_function_args
        self.min_fraction = min_fraction

    # region Properties

    @property
    def dwell_time_function(self) -> str:
        """Get or set the function used to randomly sample the desired dwell time space.

        Raises:
            ValueError: Raised when an unsupported function type is provided.
        """
        return self._dwell_time_function

    @dwell_time_function.setter
    def dwell_time_function(self, value: str):
        if value not in self._supported_functions:
            raise ValueError("{} is not a valid function.".format(value))
        self._dwell_time_function = value

    @property
    def dwell_time_function_args(self) -> tuple:
        """Get or set the dwell time space to be randomly sampled."""
        return self._dwell_time_function_args

    @dwell_time_function_args.setter
    def dwell_time_function_args(self, value):
        self._dwell_time_function_args = value

    @property
    def events_per_seed(self):
        """Get or set the number of samples to create per seed (excluding the background seed).

        Raises:
            TypeError: Raised when provided _events_per_seed value is not of type int or dict.
        """
        return self._events_per_seed

    @events_per_seed.setter
    def events_per_seed(self, value):
        if not isinstance(value, int) and not isinstance(value, dict):
            raise TypeError("Property 'events_per_seed' key must be of type 'int' or 'dict'!")

        self._events_per_seed = value

    @property
    def fwhm_function(self) -> str:
        """Get or set the function used to randomly sample the desired full-width-half-max (FWHM)
        ratio space.

        Raises:
            ValueError: Raised when an unsupported function type is provided.
        """
        return self._fwhm_function

    @fwhm_function.setter
    def fwhm_function(self, value: str):
        if value not in self._supported_functions:
            raise ValueError("{} is not a valid function.".format(value))
        self._fwhm_function = value

    @property
    def fwhm_function_args(self) -> tuple:
        """Get or set the full-width-half-max (FWHM) space to be randomly sampled."""
        return self._fwhm_function_args

    @fwhm_function_args.setter
    def fwhm_function_args(self, value):
        self._fwhm_function_args = value

    @property
    def min_fraction(self) -> float:
        """Get or set the percentage of the peak amplitude to exclude."""
        return self._min_fraction

    @min_fraction.setter
    def min_fraction(self, value: float):
        self._min_fraction = value

    @property
    def sample_interval(self) -> float:
        """Get or set the sample interval (in seconds) at which the events are simulated."""
        return self._sample_interval

    @sample_interval.setter
    def sample_interval(self, value: float):
        self._sample_interval = value

    @property
    def snr_function(self) -> str:
        """Get or set the function used to randomly sample the desired signal-to-noise
        (SNR) ratio space.

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

    # endregion

    def _calculate_passby_shape(self, fwhm: float):
        """Returns a pass-by shape with maximum of 1 which goes from min_fraction to min_fraction of
        signal with specified fwhm.

        Args:
            fwhm: Defines the full width at half maximum value to use for calculating
                the passby shape.

        Returns:
            The array of floats representing the passby shape.

        Raises:
            None.
        """
        lim = np.sqrt((1-self.min_fraction)/self.min_fraction)
        samples = np.arange(-lim, lim, self.sample_interval / fwhm / 2)
        return 1 / (np.power(samples, 2) + 1)

    def _generate_single_passby(self, fwhm: float, snr: float, dwell_time: float,
                                fg_seed: np.array, bg_seed: np.array, fg_ecal: np.array,
                                fg_sources: pd.Series, bg_sources: pd.Series):
        """Generates sampleset with a sequence of spectra representative of a single pass-by.

            A source template is scaled up and then back down over the duration of a pass-by.
            The following illustrates a general shape of pass-by's in terms of total counts.
                        ********
                      **        **
                    **            **
                   *|---- FWHM ----|*
                  *                  *
               ***                    ***
            ***                          ***
            Each sample (*) represents some number of total counts obtained over
            `dwell_time / sample_interval` seconds.

        Args:
            fwhm: full width at half maximum of the (bell-like) shape of the source count rate
                over time.
            snr: overall signal-to-noise ratio (SNR) of the passby event.
            dwell_time: overall time (seconds) the source is present.
            fg_seed: source template to use for calculating the passby; the value of each
                channel should be calculated as counts divided by total counts.
            bg_seed: background template to use for calculating the passby; the value of each
                channel should be calculated as counts divided by total counts.
            fg_ecal: The e-cal terms for the provided source seed.
            fg_sources: underlying ground truth proportions of anomalous sources.
            bg_sources: underlying ground truth proportions of background sources.

        Returns:
            A SampleSet object containing the synthesized passby data.

        Raises:
            None.
        """
        event_snr_targets = self._calculate_passby_shape(fwhm) * snr
        dwell_targets = np.zeros(int(dwell_time / self.sample_interval))
        snr_targets = np.concatenate((event_snr_targets, dwell_targets))
        n_samples = len(snr_targets)
        live_times = np.ones(n_samples) * self.sample_interval

        fg_ss, bg_ss, gross_ss = self._get_batch(
            fg_seed,
            fg_sources,
            bg_seed,
            bg_sources,
            fg_ecal,
            live_times,
            snr_targets
        )

        if self.normalize_sources:
            if self.return_fg:
                fg_ss.normalize_sources()
            if self.return_bg:
                bg_ss.normalize_sources()
            if self.return_gross:
                gross_ss.normalize_sources()

        return fg_ss, bg_ss, gross_ss

    def generate(self, fg_seeds_ss: SampleSet, bg_seeds_ss: SampleSet,
                 verbose: bool = True) \
            -> List[Tuple[SampleSet, SampleSet, SampleSet]]:
        """Generate a list of sample sets where each SampleSets represents a pass-by as a
        sequence of spectra.

        Args:
            fg_seeds_ss: spectra normalized by total counts to be used
                as the source component(s) of spectra.
            bg_seeds_ss: spectra normalized by total counts to be used
                as the background components of gross spectra.
            verbose: whether to display output from synthesis.

        Returns:
            A list of tuples of SampleSets where each tuple represents a pass-by event
            captured in any or all of three ways: foreground, background, or gross.

        Raises:
            None.
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

        args = []
        for bg_i in range(bg_seeds_ss.n_samples):
            bg_pmf = bg_seeds_ss.spectra.iloc[bg_i]
            bg_sources = bg_seeds_ss.sources.iloc[bg_i]
            fwhm_targets = get_distribution_values(self.fwhm_function,
                                                   self.fwhm_function_args,
                                                   self.events_per_seed,
                                                   self._rng)
            snr_targets = get_distribution_values(self.snr_function,
                                                  self.snr_function_args,
                                                  self.events_per_seed,
                                                  self._rng)
            dwell_time_targets = get_distribution_values(self.dwell_time_function,
                                                         self.dwell_time_function_args,
                                                         self.events_per_seed,
                                                         self._rng)
            for fg_i in range(fg_seeds_ss.n_samples):
                fg_pmf = fg_seeds_ss.spectra.iloc[fg_i]
                fg_sources = fg_seeds_ss.sources.iloc[fg_i]
                fg_ecal = fg_seeds_ss.ecal[fg_i]
                for t_i in range(self.events_per_seed):
                    fwhm = fwhm_targets[t_i]
                    snr = snr_targets[t_i]
                    dwell_time = dwell_time_targets[t_i]
                    pb_args = (fwhm, snr, dwell_time, fg_pmf, bg_pmf,
                               fg_ecal, fg_sources, bg_sources)
                    args.append(pb_args)

        # TODO: the follow prevents periodic progress reports
        passbys = [self._generate_single_passby(*a) for a in args]

        if verbose:
            delay = time() - tstart
            self._report_completion(delay)

        return passbys
