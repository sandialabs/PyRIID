# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This modules contains utilities for synthesizing gamma spectra for a source moving past a
detector.
"""
import random
from typing import Any, List

import numpy as np
import pandas as pd

from riid.data import SampleSet
from riid.data.labeling import BACKGROUND_LABEL, label_to_index_element


class PassbySynthesizer():
    """Creates synthetic pass-by events as sequences of gamma spectra."""
    _supported_functions = ["uniform", "log10", "discrete", "list"]

    def __init__(self, seeds: SampleSet, events_per_seed: int = 2, sample_interval: float = 0.125,
                 background_cps: float = 300.0, subtract_background: bool = True,
                 dwell_time_function: str = "uniform", dwell_time_function_args=(0.25, 8.0),
                 fwhm_function: str = "discrete", fwhm_function_args=(1,),
                 snr_function: str = "uniform", snr_function_args=(1.0, 10.0),
                 min_fraction: float = 0.005, random_state: int = None):
        """Constructs a synthetic passy-by generator.

        Args:
            seeds: Defines the known distributions of counts across all channels such
                that all channels sum to 1. Must contain one background seed.
            samples_per_seed: Defines the number of synthetic samples to randomly generate
                per seed.
            background_cps: Defines the constant rate of gammas from background.
            background_n_cps: Defines the constant rate of neutrons from background.
            subtract_background: Determines if generated spectra are foreground-only.
                If False, generated spectra are gross spectra (foreground + background).
            live_time_function: Defines the string that names the method of sampling
                for target live time values. Options: uniform; log10; discrete; list.
            live_time_function_args: Defines the range of values which are sampled in the
                fashion specified by the `live_time_function` argument.
            snr_function: Defines the string that names the method of sampling for target
                signal-to-noise ratio values. Options: uniform; log10; discrete; list.
            snr_function_args: Defines the range of values which are sampled in the fashion
                specified by the `snr_function` argument.
            mixture_size: Defines the number of seeds to mix together.
            mixture_min_contribution: Defines the minimum ratio of counts that a seed must
                contribute.
            random_state: Defines the random seed value used to reproduce specific data sets.

        Returns:
            None.

        Raises:
            None.
        """
        self.seeds = seeds
        self.detector_info = seeds.detector_info
        self.events_per_seed = events_per_seed
        self.sample_interval = sample_interval
        self.background_cps = background_cps
        self.subtract_background = subtract_background
        self.dwell_time_function = dwell_time_function
        self.dwell_time_function_args = dwell_time_function_args
        self.fwhm_function = fwhm_function
        self.fwhm_function_args = fwhm_function_args
        self.snr_function = snr_function
        self.snr_function_args = snr_function_args
        self.min_fraction = min_fraction
        self.random_state = random_state

    def __str__(self):
        output = "SyntheticGenerationConfig()"
        for k, v in sorted(vars(self).items()):
            output += "  {}: {}".format(k, str(v))
        return output

    def __getitem__(self, key):
        item = getattr(self, key)
        return item

    def __setitem__(self, key, value):
        setattr(self, key, value)

    # region Properties

    @property
    def background_cps(self) -> float:
        """Get or set the counts per second contributed by background radiation."""
        return self._background_cps

    @background_cps.setter
    def background_cps(self, value: float):
        self._background_cps = value

    @property
    def detector_info(self) -> dict:
        """Get or set the detector info dictionary, which includes values such as the
        unique name of the physical detector associated with the provided
        seeds. Used to determine where in the data directory to auto-cache the sample set.
        """
        return self._detector_info

    @detector_info.setter
    def detector_info(self, value: str):
        self._detector_info = value

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
    def random_state(self) -> int:
        """Get or set the seed for the random number generator. Used when trying to make
        reproducible SampleSets.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, value: int):
        self._random_state = value

    @property
    def sample_interval(self) -> float:
        """Get or set the sample interval (in seconds) at which the events are simulated."""
        return self._sample_interval

    @sample_interval.setter
    def sample_interval(self, value: float):
        self._sample_interval = value

    @property
    def seeds(self) -> SampleSet:
        """Get or set the SampleSet of non-poisson sampled gamma spectra representing the perfect
        responses given by a specific detector when observing a an isotope for an "sufficiently
        large" live time. Each seed generally represents a single source of radiation, such as
        K40, Th232, Ba133, Y88, etc., however other seeds which incorporate sources + shielding
        are perfectly valid.

        Raises:
            ValueError: Raised when a background seed value is not provided.
        """
        return self._seeds

    @seeds.setter
    def seeds(self, ss: SampleSet):
        labels = ss.get_labels().values
        if BACKGROUND_LABEL not in labels:
            raise ValueError(f"A seed with the label '{BACKGROUND_LABEL}' must be provided.")
        self._seeds = ss

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

    @property
    def subtract_background(self) -> bool:
        """Get or set the flag for whether or not to include counts from background
        in the final spectra.
        """
        return self._subtract_background

    @subtract_background.setter
    def subtract_background(self, value: bool):
        self._subtract_background = value

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
                                seed_pdf: np.array, background_pdf: np.array, source: str):
        """Generates sampleset with a sequence of spectra representative of a single pass-by event.

        Args:
            fwhm: Defines the full width at half maximum value to use for calculating the passby.
            snr: Defines the signal to noise ration to use for calculating the passby.
            dwell_time: Defines the dwell time for the source to use for calculating the passby.
            seed_pdf: Defines the gross seed array to use for calculating the passby; the seed_pdf
                values are calculated as a percent, e.g. counts_in_channel/total_counts.
            background_pdf: Defines the background seed array to use for calculating the passby; the
                background_pdf values are calculated as a percent, e.g.
                counts_in_channel/total_counts.
            source: Defines the string name of the source present in the passby.

        Returns:
            A SampleSet object containing the synthesized passby data.

        Raises:
            None.
        """
        event_snr_targets = self._calculate_passby_shape(fwhm) * snr
        n_event_spectra = len(event_snr_targets)
        dwell_targets = np.zeros(int(dwell_time / self.sample_interval))
        snr_targets = np.concatenate((event_snr_targets, dwell_targets))

        n_samples = len(snr_targets)
        live_times = np.ones(n_samples)

        bg_counts_expected = self.background_cps * live_times
        bg_spectra = np.random.poisson(background_pdf * bg_counts_expected[:, None])
        bg_counts = bg_spectra.sum(axis=1)

        fg_counts_expected = self.background_cps * snr_targets * live_times
        fg_spectra = np.random.poisson(seed_pdf * fg_counts_expected[:, None])
        fg_counts = fg_spectra.sum(axis=1)
        net_spectra = bg_spectra + fg_spectra
        total_counts = bg_counts + fg_counts
        excess_from_expected = total_counts - bg_counts_expected
        snrs = excess_from_expected / np.sqrt(bg_counts_expected)

        source_data = np.hstack(
            (np.full([n_samples, 1], 1))
        )

        col_level_idx = SampleSet.SOURCES_MULTI_INDEX_NAMES.index("Isotope")
        source_columns = pd.MultiIndex.from_tuples(
            [label_to_index_element(source)],
            names=SampleSet.SOURCES_MULTI_INDEX_NAMES[:col_level_idx+1]
        )
        sources = pd.DataFrame(columns=source_columns, data=source_data)
        ecal_vals = np.zeros((n_samples), dtype=float)

        info = pd.DataFrame(
                data=np.vstack((
                    live_times,
                    total_counts,
                    snrs,
                    ecal_vals,
                    ecal_vals,
                    ecal_vals,
                    ecal_vals,
                    ecal_vals)
                ).T,
                columns=[
                    "live_time",
                    "total_counts",
                    "snr",
                    *SampleSet.ECAL_INFO_COLUMNS
                ]
            )

        if self.subtract_background:
            spectra = net_spectra - bg_spectra
        else:
            spectra = net_spectra

        ss = SampleSet()
        ss.spectra = pd.DataFrame(spectra)

        info.update(ss.info)
        ss.info = info
        ss.sources = sources
        ss.synthesis_info = {
            "dwell_time": dwell_time,
            "event_length": n_event_spectra,
            "fwhm": fwhm,
            "snr": snr,
            "source": source,
            "total_length": n_samples,
        }
        ss.detector_info.update(self.detector_info)
        ss.measured_or_synthetic = "synthetic"

        return ss

    def _get_distribution_values(self, function: str, function_args: Any, n_values: int):
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
        if function not in self._supported_functions:
            raise ValueError("{} is not a valid function.".format(function))

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

    def _get_dwell_time_targets(self, n_samples: int) -> list:
        """Obtains a list of random dwell time target values.

        Args:
            n_samples: Defines the number of samples for which the
                dwell_time_targets should be calculated.

        Returns:
            A list containing the results of the call to
                _get_distribution_values.

        Raises:
            None.
        """
        return self._get_distribution_values(
            self.dwell_time_function,
            self.dwell_time_function_args,
            n_samples
        )

    def _get_fwhm_targets(self, n_samples: int) -> list:
        """Obtains a list of random full-width-half-max (FWHM) target values.

        Args:
            n_samples: Defines the number of samples for which the
                fwhm_targets should be calculated.
        Returns:
            A list containing the results of the call to
                _get_distribution_values.
        """
        return self._get_distribution_values(
            self.fwhm_function,
            self.fwhm_function_args,
            n_samples
        )

    def _get_snr_targets(self, n_samples) -> list:
        """Obtains a list of random SNR target values.

        Args:
            n_samples: Defines the number of samples for which the
                fwhm_targets should be calculated.

        Returns:
            A list containing the results of the call to
                _get_distribution_values.

        Raises:
            None.
        """
        return self._get_distribution_values(
            self.snr_function,
            self.snr_function_args,
            n_samples
        )

    def generate(self) -> List[SampleSet]:
        """Generate a list of sample sets where each SampleSets represents a pass-by as a
        sequence of spectra.

        Returns:
            A list of SampleSets where each SampleSet represents a pass-by event.

        Raises:
            None.
        """
        if self.random_state:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        labels = self.seeds.get_labels()
        isotope_seeds = self.seeds[labels != BACKGROUND_LABEL]
        isotope_seeds_labels = isotope_seeds.get_labels()
        background_seed = self.seeds[labels == BACKGROUND_LABEL]
        bs_indices = np.full(background_seed.n_samples, False)
        bs_indices[0] = True
        background_seed = background_seed[bs_indices]
        background_spectrum = background_seed.spectra.iloc[0, :].values
        background_pdf = background_spectrum.clip(0)
        background_pdf = background_pdf / background_pdf.sum()

        # Generate samples for each seed
        args = []
        for i, _ in enumerate(isotope_seeds.spectra.index):
            seed_spectrum = isotope_seeds.spectra.iloc[i, :].values
            source = isotope_seeds_labels[i]
            seed_pdf = seed_spectrum.clip(0)
            seed_pdf = np.array(seed_pdf) / seed_pdf.sum()
            n_samples = self.events_per_seed
            if isinstance(self.events_per_seed, dict):
                n_samples = self.events_per_seed[source]
            fwhm_targets = self._get_fwhm_targets(n_samples)
            snr_targets = self._get_snr_targets(n_samples)
            dwell_time_targets = self._get_dwell_time_targets(n_samples)
            for fwhm, snr, dwell in zip(fwhm_targets, snr_targets, dwell_time_targets):
                args.append((fwhm, snr, dwell, seed_pdf, background_pdf, source))

        events = [self._generate_single_passby(*a) for a in args]
        return events
