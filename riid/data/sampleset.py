# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains the `SampleSet` class and other `SampleSet`-related functions."""
from __future__ import \
    annotations  # Enables SampleSet hints inside SampleSet itself

import copy
import json
import logging
import operator
import random
import re
import warnings
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Tuple, Union

import numpy as np
import pandas as pd
import tqdm
from scipy import stats
from scipy.interpolate import interp1d
from scipy.spatial import distance

import riid
from riid.data import get_expected_spectra
from riid.data.labeling import (NO_CATEGORY, NO_ISOTOPE, NO_SEED,
                                _find_category, _find_isotope)
from riid.gadras.pcf import (_dict_to_pcf, _pack_compressed_text_buffer,
                             _pcf_to_dict, _unpack_compressed_text_buffer)


class SpectraState(int, Enum):
    """States in which SampleSet spectra can exist."""
    Unknown = 0
    Counts = 1
    L1Normalized = 2
    L2Normalized = 3


class SpectraType(int, Enum):
    """Types for SampleSet spectra."""
    Unknown = 0
    Background = 1
    Foreground = 2
    Gross = 3
    BackgroundForeground = 4


class SampleSet():
    """A collection of spectrum samples and their metadata."""
    DEAD_TIME_PROP_INFO_KEY = "dead_time_prop"
    SOURCES_MULTI_INDEX_NAMES = (
        "Category",
        "Isotope",
        "Seed"
    )
    ECAL_INFO_COLUMNS = (
        "ecal_order_0",
        "ecal_order_1",
        "ecal_order_2",
        "ecal_order_3",
        "ecal_low_e"
    )
    DEFAULT_INFO_COLUMNS = (
        "description",
        "timestamp",
        "live_time",
        "real_time",
        "total_counts",
        "snr",
        "neutron_counts",
        "distance_cm",
        *ECAL_INFO_COLUMNS,
        "areal_density",
        "atomic_number",
        "occupancy_flag",
        "tag",
    )
    DEFAULT_BG_SEED_NAMES = [
        "Cosmic",
        "PotassiumInSoil",
        "UraniumInSoil",
        "ThoriumInSoil",
    ]
    SUPPORTED_STATES_FOR_ARITHMETIC = (
        SpectraState.Counts.value,
        SpectraState.L1Normalized.value
    )
    DEFAULT_EXCLUSIONS_FROM_COMPARISON = (
        "timestamp",
        "description",
        "occupancy_flag",
        "tag",
    )

    def __init__(self):
        """
        Expected sizes for DataFrames:
            self._spectra: [n_samples, n_channels]
            self._sources: [n_samples, n_sources]
        """
        self._spectra = pd.DataFrame()
        self._sources = pd.DataFrame()
        self._info = pd.DataFrame(columns=SampleSet.DEFAULT_INFO_COLUMNS)
        self._detector_info = {}
        self._synthesis_info = {}
        self._prediction_probas = pd.DataFrame()
        self._measured_or_synthetic = None
        self.pyriid_version = riid.__version__
        self._spectra_state = SpectraState.Unknown
        self._spectra_type = SpectraType.Unknown
        self._classified_by = ""

    def __bool__(self):
        return bool(len(self))

    def __getitem__(self, key: Union[slice, int, list]):
        selection = key
        if isinstance(key, int):
            selection = slice(key, key+1)
        elif isinstance(key, pd.Series) and key.dtype == bool:
            selection = self.spectra.index[key]

        sub_ss = copy.copy(self)
        sub_ss.spectra = sub_ss.spectra.iloc[selection].reset_index(drop=True)
        sub_ss.sources = sub_ss.sources.iloc[selection].reset_index(drop=True)
        sub_ss.info = sub_ss.info.iloc[selection].reset_index(drop=True)
        if not sub_ss.prediction_probas.empty:
            sub_ss.prediction_probas = sub_ss.prediction_probas.iloc[selection] \
                .reset_index(drop=True)

        return sub_ss

    def __len__(self):
        return self.n_samples

    def __str__(self):
        UNKNOWN_VALUE = "Unknown"
        detector_info_str = self.detector_info if self.detector_info else UNKNOWN_VALUE
        sources_present_str = ", ".join(sorted(np.unique(self.get_labels()))) \
            if not self.sources.empty \
            else UNKNOWN_VALUE
        predictions_str = ", ".join(sorted(np.unique(self.get_labels()))) \
            if not self.prediction_probas.empty \
            else "None"
        pyriid_version_str = self.pyriid_version if self.pyriid_version else UNKNOWN_VALUE
        info_dict = {
            "# of samples": self.n_samples,
            "# of channels": self.n_channels,
            "Detector": detector_info_str,
            "Present predictions": predictions_str,
            "Present sources": sources_present_str,
            "PyRIID version": pyriid_version_str,
        }
        summary = "\n".join(_dict_to_bulleted_list(info_dict))
        return f"SampleSet Summary:\n{summary}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, ss: SampleSet):
        return np.allclose(self._spectra.values, ss._spectra.values, atol=1e-3)

    def _check_arithmetic_supported(self, ss2: SampleSet):
        if ss2.n_samples != 1 and ss2.n_samples != self.n_samples:
            n_sample_str = (
                "You can only add/subtract SampleSet objects "
                "when the second SampleSet has only one spectrum "
                "(the spectrum will be scaled to match each sample of the first SampleSet) or "
                "when the second SampleSet has the same number of samples as the first "
                "(no rescaling will occur)."
            )
            raise InvalidSampleCountError(n_sample_str)
        if self.n_channels != ss2.n_channels:
            channel_str = f"({self.n_channels} != {ss2.n_channels})"
            raise ChannelCountMismatchError(f"Mismatched spectra channels {channel_str}!")
        if self.spectra_state != ss2.spectra_state:
            state_str = f"({self.spectra_state} != {ss2.spectra_state})"
            raise SpectraStateMismatchError(f"Mismatched spectra states {state_str}!")
        if self.spectra_state.value not in self.SUPPORTED_STATES_FOR_ARITHMETIC:
            raise ValueError(f"{self.spectra_state} spectra state not supported for arithmetic!")
            # Don't need to check spectra state of ss2 since they passed the prior equality check

    def _get_scaled_spectra(self, ss: SampleSet) -> np.ndarray:
        spectra_in_counts = ss.spectra.iloc[0].values.copy()
        if ss.spectra_state == SpectraState.L1Normalized:
            spectra_in_counts *= ss.info.iloc[0].total_counts
        live_times = ss.info.iloc[0].live_time
        spectra_in_cps = spectra_in_counts / live_times
        scaled_spectra = np.concatenate(
            spectra_in_cps * self.info.live_time.values[:, np.newaxis, np.newaxis]
        )
        return scaled_spectra

    def _get_arithmetic_result(self, ss: SampleSet, op) -> SampleSet:
        self._check_arithmetic_supported(ss)

        if ss.n_samples == 1:
            scaled_spectra = self._get_scaled_spectra(ss)
        else:
            scaled_spectra = ss.spectra.values  # Assumed to already be scaled
        new_ss = self[:]

        is_l1_normalized = new_ss.spectra_state == SpectraState.L1Normalized
        if is_l1_normalized:
            new_ss.spectra = new_ss.spectra.multiply(new_ss.info.total_counts, axis=0)

        new_ss.spectra = op(new_ss.spectra, scaled_spectra)

        if is_l1_normalized:
            new_ss.normalize()

        return new_ss

    def __add__(self, bg_ss: SampleSet) -> SampleSet:
        """Add the given background spectr(um|a) to the spectra of the current SampleSet.
        """
        if bg_ss.spectra_type.value != SpectraType.Background.value:
            msg = (
                "`bg_ss` argument must have a `spectra_type` of `Background`. "
                f"Its `spectra_type` is `{bg_ss.spectra_type}`."
            )
            raise ValueError(msg)
        if self.spectra_type.value != SpectraType.Gross.value and \
           self.spectra_type.value != SpectraType.Foreground.value:
            msg = (
                "Current `SampleSet` must have a `spectra_type` of `Gross` or `Foreground`. "
                f"Current `spectra_type` is `{self.spectra_type}`."
            )
            raise ValueError(msg)

        new_ss = self._get_arithmetic_result(bg_ss, operator.add)
        new_ss.spectra_type = SpectraType.Gross
        return new_ss

    def __sub__(self, ss: SampleSet) -> SampleSet:
        """Subtract the given background or foreground spectr(um|a) from the spectra of the current
        `SampleSet`.
        """
        if ss.spectra_type.value != SpectraType.Background.value and \
           ss.spectra_type.value != SpectraType.Foreground.value:
            msg = (
                "`ss` argument must have a `spectra_type` of `Background` of `Foreground`. "
                f"Its `spectra_type` is `{ss.spectra_type}`."
            )
            raise ValueError(msg)
        if self.spectra_type.value != SpectraType.Gross.value:
            msg = (
                "Current `SampleSet` must have a `spectra_type` of `Gross`. "
                f"Current `spectra_type` is `{self.spectra_type}`."
            )
            raise ValueError(msg)

        new_ss = self._get_arithmetic_result(ss, operator.sub)
        if ss.spectra_type == SpectraType.Background:
            new_ss.spectra_type = SpectraType.Foreground
        else:
            new_ss.spectra_type = SpectraType.Background
        return new_ss

    # region Properties

    @property
    def category_names(self):
        """Get the names of the categories involved in this SampleSet."""
        return self.sources.columns.get_level_values("Category")

    @property
    def classified_by(self):
        """Get or set the UUID of the model that classified the SampleSet.

        TODO: Implement as third dimension to `prediction_probas` DataFrame
        """
        return self._classified_by

    @classified_by.setter
    def classified_by(self, value):
        self._classified_by = value

    @property
    def detector_info(self):
        """Get or set the detector info on which this `SampleSet` is based.

        TODO: implement as DataFrame
        """
        return self._detector_info

    @detector_info.setter
    def detector_info(self, value):
        self._detector_info = value

    @property
    def difficulty_score(self, mean=10.0, std=3.0) -> float:
        """Compute the "difficulty" of the `SampleSet` on a scale of 0 to 1,
        where 0 is easiest and 1 is hardest.

        The difficulty of a `SampleSet` is the mean of the individual sample difficulties.
        Each sample's difficulty is determined by where its signal strength (SNR)
        falls on the survival function (AKA reliability function, or 1 - CDF) for a
        logistic distribution.
        The decision to use this distribution is based on empirical data showing that
        classifier performance increases in a logistic fashion as signal strength increases.

        The primary use of this function is for situations where ground truth is unknown
        and you want to get an idea of how "difficult" it will be for a model
        to make predictions.  Consider the following example:
            Given a `SampleSet` for which ground truth is unknown and the signal strength for
            each spectrum is low.
            A model considering, say, 100+ isotopes will see this `SampleSet` as quite
            difficult, whereas a model considering 5 isotopes will, being more specialized,
            see the `SampleSet` as easier.
            Of course, this makes the assumption that both models were trained to the
            isotope(s) contained in the test `SampleSet`.

        Based on the previous example, the unfortunate state of this function is that you
        must know how to pick means and standard deviations which properly reflect the:

        - number of target isotopes
        - amount of variation within each isotope (i.e., shielding, scattering, etc.)
        - detector resolution

        A future goal of ours is to provide updates to this function and docstring which make
        choosing the mean and standard deviation easier/automated based on more of our findings.

        Arguments:
            mean: SNR value representing the mean of the logistic distribution
            std: standard deviation of the logistic distribution

        Returns:
            The mean of all SNR values passed through a logistic survival function
        """
        snrs: np.ndarray = self.info.snr.clip(1e-6)
        score = float(stats.logistic.sf(snrs, loc=mean, scale=std).mean())

        return score

    @property
    def ecal(self):
        """Get or set the ecal terms."""
        ecal_terms = self.info[list(self.ECAL_INFO_COLUMNS)].to_numpy(dtype=float)
        return ecal_terms

    @ecal.setter
    def ecal(self, value):
        self.info.loc[:, self.ECAL_INFO_COLUMNS[0]] = value[0]
        self.info.loc[:, self.ECAL_INFO_COLUMNS[1]] = value[1]
        self.info.loc[:, self.ECAL_INFO_COLUMNS[2]] = value[2]
        self.info.loc[:, self.ECAL_INFO_COLUMNS[3]] = value[3]
        self.info.loc[:, self.ECAL_INFO_COLUMNS[4]] = value[4]

    @property
    def info(self):
        """Get or set the info DataFrame."""
        return self._info

    @info.setter
    def info(self, value):
        self._info = value

    @property
    def isotope_names(self):
        """Get the names of the isotopes involved in this SampleSet."""
        return self.sources.columns.get_level_values("Isotope")

    @property
    def measured_or_synthetic(self):
        """Get or set the value for measured_or_synthetic for the SampleSet object."""
        return self._measured_or_synthetic

    @measured_or_synthetic.setter
    def measured_or_synthetic(self, value):
        self._measured_or_synthetic = value

    @property
    def n_channels(self):
        """Get the number of channels included in the spectra DataFrame.
        Channels may also be referred to as bins.
        """
        return self._spectra.shape[1]

    @property
    def n_samples(self):
        """Get the number of samples included in the spectra dataframe,
        where each row is a sample.
        """
        return self._spectra.shape[0]

    @property
    def prediction_probas(self):
        """Get or set the prediction_probas DataFrame."""
        return self._prediction_probas

    @prediction_probas.setter
    def prediction_probas(self, value):
        if isinstance(value, pd.DataFrame):
            self._prediction_probas = value
        else:
            self._prediction_probas = pd.DataFrame(value, columns=self.sources.columns)

    @property
    def seed_names(self):
        """Get the names of the seeds involved in this `SampleSet`."""
        return self.sources.columns.get_level_values("Seed")

    @property
    def sources(self):
        """Get or set the sources `DataFrame`."""
        return self._sources

    @sources.setter
    def sources(self, value):
        self._sources = value.replace(np.nan, 0)

    @property
    def spectra(self):
        """Get or set the spectra `DataFrame`."""
        return self._spectra

    @spectra.setter
    def spectra(self, value):
        self._spectra = value

    @property
    def spectra_state(self: SpectraState):
        """Get or set the spectra state."""
        return self._spectra_state

    @spectra_state.setter
    def spectra_state(self, value: SpectraState):
        self._spectra_state = value

    @property
    def spectra_type(self: SpectraType):
        """Get or set the spectra type."""
        return self._spectra_type

    @spectra_type.setter
    def spectra_type(self, value: SpectraType):
        self._spectra_type = value

    @property
    def synthesis_info(self):
        """Get or set the information that was used to by a synthesizer."""
        return self._synthesis_info

    @synthesis_info.setter
    def synthesis_info(self, value):
        self._synthesis_info = value

    # endregion

    def _channels_to_energies(self, fractional_energy_bins,
                              offset: float, gain: float, quadratic: float, cubic: float,
                              low_energy: float) -> np.ndarray:
        r"""Convert fractional energy bins to the energy represented by the lower edge of the bin.

        A spectrum represents a range of energy (e.g., 0 keV to 3000 keV).
        The energy range represented by a spectrum can be inferred from its energy calibration
        (e-cal for short), if known.  Using the convention in ANSI N42.42-2006, the e-cal is
        defined using the following terms:

        - \( a_0 \): offset (ecal_order_0)
        - \( a_1 \): gain (ecal_order_1)
        - \( a_2 \): quadratic (ecal_order_2)
        - \( a_3 \): cubic (ecal_order_3)
        - \( a_4 \): low energy (ecal_low_e)

        The energy represented by the lower edge of channel \( i \) is then calculated as follows:

        $$
        E_i = a_0 + a_1 * x + a_2 * x^2 + a_3 * a^3 + \frac{a_4}{1 + 60x}
        $$

        where \( x \) is the fractional energy of the spectral range.
        Also, under this scheme and ignoring the non-linear terms (e.g., \( a_2 \) through
        \( a_4 \)), the range of energy represented by a spectrum can be quickly inferred
        from the e-cal as:

        - `min_energy = offset`
        - `max_energy = gain + offset`

        Note: spline interpolation is another way to characterize the energy calibration of a
        spectrum. In GADRAS, this is accomplished via deviation pairs; PyRIID does not currently
        support this but may in the future.

        Args:
            fractional_energy_bins: array of fractions representing the spacing of the channels
            offset: e-cal order 0 term
            gain: e-cal order 1 term
            quadratic: e-cal order 2 term
            cubic: e-cal order 3 term
            low_energy: e-cal low energy term

        Returns:
            Array containing the energies represented by the lower edge of each channel
            (hape depends on the shape of input values)
        """
        channel_energies = \
            offset + \
            fractional_energy_bins * gain + \
            fractional_energy_bins**2 * quadratic + \
            fractional_energy_bins**3 * cubic + \
            low_energy / (1 + 60 * fractional_energy_bins)
        return channel_energies

    def _check_target_level(self, target_level,
                            levels_allowed=SOURCES_MULTI_INDEX_NAMES):
        if target_level not in levels_allowed:
            raise ValueError((
                f"'{target_level}' is not an appropriate target level. "
                f"Acceptable values are: {levels_allowed}."
            ))

    def _get_dead_time_proportions(self):
        live_times = self.info.live_time.values
        real_times = self.info.real_time.values
        dead_time_props = (1.0 - live_times / real_times)
        dead_time_props = np.nan_to_num(dead_time_props, nan=1.0)
        return dead_time_props

    def all_spectra_sum_to_one(self, rtol: float = 0.0, atol: float = 1e-4) -> bool:
        """Checks if all spectra are normalized to sum to one."""
        spectra_counts = self.spectra.sum(axis=1).values
        all_sum_to_one = np.all(np.isclose(spectra_counts, 1, rtol=rtol, atol=atol))
        return all_sum_to_one

    def as_ecal(self, new_offset: float, new_gain: float,
                new_quadratic: float, new_cubic: float,
                new_low_energy: float) -> SampleSet:
        """Re-bin spectra based on energy by interpolating the current shape from the current
        binning structure to a new one.

        Warning: this operation is likely to add or remove spectral information
        depending on the new energy calibration values used.

        Args:
            new_offset: new gain value, i.e. the 0-th e-cal term
            new_gain: new gain value, i.e. the 1-th e-cal term
            new_quadratic: new quadratic value, i.e. the 2-th e-cal term
            new_cubic: new cubic value, i.e. the 3-th e-cal term
            new_low_energy: new low energy term

        Returns:
            A new `SamleSet` with `spectra` and `info` DataFrames

        Raises:
            `ValueError` when no argument values are provided
        """
        new_args = [new_offset, new_gain, new_quadratic, new_cubic, new_low_energy]
        if all(v is None for v in new_args):
            raise ValueError("At least one argument value must be provided.")

        ecal_cols = list(self.ECAL_INFO_COLUMNS)
        new_ecal = self.info[ecal_cols].copy()
        new_ecal.ecal_order_0 = new_offset
        new_ecal.ecal_order_1 = new_gain
        new_ecal.ecal_order_2 = new_quadratic
        new_ecal.ecal_order_3 = new_cubic
        new_ecal.ecal_low_e = new_low_energy

        all_original_channel_energies = self.get_all_channel_energies()
        all_new_channel_energies = []
        fractional_energy_bins = np.linspace(0, 1, self.n_channels)
        for i in range(self.n_samples):
            offset, gain, quadratic, cubic, low_energy = new_ecal.iloc[i].values
            new_channel_energies = self._channels_to_energies(
                fractional_energy_bins,
                offset, gain, quadratic, cubic, low_energy
            )
            all_new_channel_energies.append(new_channel_energies)

        new_spectra = np.zeros([self.n_samples, self.n_channels])
        # Perform interpolation
        for i in range(self.n_samples):
            f = interp1d(
                all_original_channel_energies[i],
                self.spectra.values[i],
                kind="slinear",
                fill_value=0,
                bounds_error=False
            )
            new_spectrum = f(all_new_channel_energies[i])
            new_spectra[i] = new_spectrum

        # Update spectra and info DataFrames
        new_ss = self[:]
        new_ss.spectra = pd.DataFrame(new_spectra)
        new_ss.info.total_counts = new_ss.spectra.sum(axis=1)
        new_ss.info[ecal_cols] = new_ecal
        return new_ss

    def as_regions(self, rois: list) -> SampleSet:
        """Obtains a new `SampleSet` where the spectra are limited to specific
        regions of interest (ROIs).

        Notes:
            - If your samples have disparate energy calibration terms, call `as_ecal()` first
              to align channel space, then you may call this function. Otherwise, it is possible
              to end up with a ragged array of spectra, which we do not support.
            - After this call, `spectra` will have columns filled in with energy values for
              convenience. As such, in the context of the returned `SampleSet`, the energy
              calibration terms in `info` will no longer have any meaning, and any subsequent
              calls to methods like `as_ecal()` would not make sense.  This method is intended
              as a last step to be performed right before analysis of whatever kind.

        Args:
            rois: a list of 2-tuples where tuple represents (low energy, high energy)

        Returns:
            A new `SamleSet` with only ROIs remaining in the `spectra` DataFrame

        Raises:
            `ValueError` when no argument values are provided
        """
        if not rois:
            raise ValueError("At least one ROI must be provided.")
        all_ecals = self.ecal
        all_ecals_are_same = np.isclose(all_ecals, all_ecals[0]).all()
        if not all_ecals_are_same:
            msg = "Spectra have different energy calibrations; consider `as_ecal()` first."
            raise ValueError(msg)

        energies = self.get_channel_energies(0)
        mask = _get_energy_roi_masks(rois, energies)
        new_spectra = self.spectra.to_numpy(dtype=float)[:, mask]
        new_spectra = new_spectra.reshape((self.n_samples, -1))
        mask_energies = energies[mask]

        new_ss = self[:]
        new_ss.spectra = pd.DataFrame(new_spectra, columns=mask_energies)
        new_ss.info.total_counts = new_ss.spectra.sum(axis=1)
        return new_ss

    def check_seed_health(self, dead_time_threshold=1.0):
        """Checks health of all spectra and info assuming they are seeds.

        Invalidate states for which we currently check:

        - spectra do not sum to 1
        - dead time greater than or equal to provided threshold

        Args:
            dead_time_threshold: value at which seed dead time is unacceptable

        Raises:
            `AssertionError` if any check fails
        """
        all_spectra_sum_to_one = self.all_spectra_sum_to_one()
        assert all_spectra_sum_to_one

        if self.DEAD_TIME_PROP_INFO_KEY not in self.info.columns:
            self.set_dead_time_proportions()
        dead_samples = self.info[self.DEAD_TIME_PROP_INFO_KEY] >= dead_time_threshold
        all_samples_are_alive = not np.any(dead_samples)
        assert all_samples_are_alive

    def clip_negatives(self, min_value: float = 0):
        """Clip spectrum values to some minimum value.

        Args:
            min_value: value to which to clip existing spectrum values
        """
        self._spectra = pd.DataFrame(data=self._spectra.clip(min_value))

    def compare_to(self, ss: SampleSet, n_bins: int = 20, density: bool = False) \
            -> Tuple[dict, dict, dict]:
        """Compare the current `SampleSet` to another `SampleSet`.

        The function only compares the selected, mutual information of
        each `SampleSet` by computing the Jensen-Shannon distance between
        histograms of that information.

        Args:
            ss: `SampleSet` to compare to
            n_bins: number of bins we will sort both sample sets by
            density: whether histograms should be in counts or density

        Returns:
            Tuple of the following:

            - ss1_stats: dictionary of stats describing the first `SampleSet`
            - ss2_stats: dictionary of stats describing the second `SampleSet`
            - col_comparisons: dictionary of distance values comparing each stat
        """
        TARGET_SUMMARY_STATS = ["min", "max", "median", "mean", "std"]
        STAT_PRECISION = 3

        # Get info columns we want to compare
        info_columns = [x for x in list(self.DEFAULT_INFO_COLUMNS)
                        if x not in list(self.DEFAULT_EXCLUSIONS_FROM_COMPARISON)]

        # Get both sample sets comparable columns of data
        info_df1 = self.info[info_columns]
        info_df2 = ss.info[info_columns]

        # Check valid columns in each sample set (cannot have None or 0)
        ss1_valid_cols = [
            c for c in info_df1.columns
            if pd.to_numeric(info_df1[c], errors="coerce").notnull().all() and any(info_df1[c])
        ]
        ss2_valid_cols = [
            c for c in info_df2.columns
            if pd.to_numeric(info_df2[c], errors="coerce").notnull().all() and any(info_df2[c])
        ]

        # Remove non shared column lists
        for i in ss1_valid_cols:
            if i not in ss2_valid_cols:
                ss1_valid_cols.remove(i)
        for i in ss2_valid_cols:
            if i not in ss1_valid_cols:
                ss2_valid_cols.remove(i)

        # Remove non shared column data
        info_df1 = info_df1[ss1_valid_cols]
        info_df2 = info_df2[ss2_valid_cols]

        # Bin the data
        ss1_stats = {}
        ss2_stats = {}
        col_comparisons = {}
        for i in ss1_valid_cols:
            ss1_stats[i] = {}
            ss2_stats[i] = {}
            hist_range = (
                min(min(info_df1[i]), min(info_df2[i])),
                max(max(info_df1[i]), max(info_df2[i]))
            )
            # Get data in bins and get counts for each bin
            hist1, bins1 = np.histogram(info_df1[i], bins=n_bins, range=hist_range, density=density)
            hist2, bins2 = np.histogram(info_df2[i], bins=n_bins, range=hist_range, density=density)
            ss1_stats[i]["density"] = density
            ss2_stats[i]["density"] = density
            ss1_stats[i]["hist"] = hist1
            ss2_stats[i]["hist"] = hist2
            ss1_stats[i]["bins"] = bins1
            ss2_stats[i]["bins"] = bins2
            ss1_stats[i]["range"] = hist_range
            ss2_stats[i]["range"] = hist_range

            summary_stats_df1 = info_df1[i].agg(TARGET_SUMMARY_STATS).round(decimals=STAT_PRECISION)
            ss1_stats[i].update(summary_stats_df1.to_dict())
            summary_stats_df2 = info_df2[i].agg(TARGET_SUMMARY_STATS).round(decimals=STAT_PRECISION)
            ss2_stats[i].update(summary_stats_df2.to_dict())

            js_dist = distance.jensenshannon(hist1, hist2, axis=0)
            col_comparisons[i] = js_dist

        return ss1_stats, ss2_stats, col_comparisons

    def concat(self, ss_list: list):
        """Vertically concatenate multiple `SampleSet`s into one `SampleSet`.

        Combining of non-DataFrame properties (e.g., detector_info, synthesis_info, etc.)
        is NOT performed - it is the responsiblity of the user to ensure proper bookkeeping for
        these properties when concatenating `SampleSet`s.

        This method is most applicable to `SampleSet`s containing measured data from the same
        detector which could not be made as a single `SampleSet` because of how measurements had to
        be taken (i.e., measurements were taken by distinct processes separated by time).
        Therefore, the user should avoid concatenating `SampleSet`s when:

        - the data is from different detectors
        (note that `SampleSet`s are given to models for prediction, and models should be 1-to-1
        with physical detectors);
        - a mix of synthetic and measured data would occur (this could make an existing
        `synthesis_info` value ambiguous, but one way to handle this would be to add a new column
        to `info` distinguishing between synthetic and measured on a per-sample basis).

        Args:
            ss_list: list of `SampleSet`s to concatenate

        Returns:
            `SampleSet` object
        """
        if not ss_list:
            return
        if not isinstance(ss_list, list) and not isinstance(ss_list, tuple):
            ss_list = [ss_list]
        spectra_types = list(
            set([self.spectra_type.value] + [x.spectra_type.value for x in ss_list])
        )
        n_spectra_types = len(spectra_types)
        if n_spectra_types == 2:
            if SpectraType.Background.value in spectra_types and \
               SpectraType.Foreground.value in spectra_types:
                self.spectra_type = SpectraType.BackgroundForeground
            else:
                self.spectra_type = SpectraType.Unknown
        elif n_spectra_types == 0 or n_spectra_types > 2:
            self.spectra_type = SpectraType.Unknown

        self._spectra = pd.concat(
            [self._spectra] + [ss.spectra for ss in ss_list],
            ignore_index=True,
            sort=False
        )
        self._sources = pd.concat(
            [self._sources] + [ss.sources for ss in ss_list],
            ignore_index=True,
            sort=False
        )
        self._sources = self._sources.where(pd.notnull(self._sources), 0)
        existing_info_df = self._info if not self._info.empty else None
        self._info = pd.concat(
            [existing_info_df] + [ss.info for ss in ss_list],
            ignore_index=True,
            sort=False
        )
        self._info = self._info.where(pd.notnull(self._info), None)
        self._prediction_probas = pd.concat(
            [self._prediction_probas] + [ss.prediction_probas for ss in ss_list],
            ignore_index=True,
            sort=False
        )
        self._prediction_probas = self._prediction_probas.where(
            pd.notnull(self._prediction_probas),
            0
        )

    def downsample_spectra(self, target_bins: int = 128, min_frac=1e-8):
        """Uniformly down-bin spectra.

        Args:
            target_bins: number of channels to which to bin the existing spectra

        Raises:
            `ValueError` when binning is not a multiple of the target binning
        """
        if self.n_channels % target_bins == 0:
            n_per_group = int(self.n_channels / target_bins)
            transformation = np.zeros([target_bins, self.n_channels])
            for i in range(target_bins):
                transformation[i, (i * n_per_group):((i * n_per_group) + n_per_group)] = 1
        else:
            msg = "Current binning ({}) is not a multiple of the target binning ({})".format(
                self.n_channels,
                target_bins
            )
            raise ValueError(msg)
        self._spectra = pd.DataFrame(
            data=np.matmul(
                self._spectra.values,
                transformation.T
            )
        )
        self._spectra[self._spectra < min_frac] = 0

    def drop_sources(self, col_names: Iterable = DEFAULT_BG_SEED_NAMES,
                     normalize_sources: bool = True,
                     target_level: str = "Seed"):
        """Drop columns from `sources`.

        Args:
            col_names: names of the sources columns to be dropped.
                The names of background seeds are used by default as removing the
                ground truth for background sources when dealing with synthetic,
                gross spectra is a common operation.
            normalize_sources: whether to normalize the sources DataFrame after dropping
                the column(s)
            target_level: `SampleSet.sources` column level to use
        """
        self.sources.drop(col_names, axis=1, inplace=True, level=target_level)
        if normalize_sources:
            self.normalize_sources()

    def drop_sources_columns_with_all_zeros(self):
        """Remove columns from the sources `DataFrame` that contain only zeros.

        Modifications are made in-place.
        """
        idxs = (self.sources != 0).any(axis=0)
        self.sources = self.sources.loc[:, idxs]
        self.sources.columns = self.sources.columns.remove_unused_levels()

        return idxs

    def drop_spectra_with_no_contributors(self) -> np.ndarray:
        """Remove samples where the spectrum has no recorded contributor.

        Modifications are inplace.

        Returns:
            A boolean mask of which rows were kept
        """
        idxs = np.where(self.sources.values.sum(axis=1) == 0)[0]
        kept_mask = ~np.in1d(self.sources.index.values, idxs)

        self.spectra = self.spectra.drop(index=idxs).reset_index(drop=True)
        self.sources = self.sources.drop(index=idxs).reset_index(drop=True)
        self.info = self.info.drop(index=idxs).reset_index(drop=True)

        return kept_mask

    def extend(self, spectra: Union[dict, list, np.array, pd.DataFrame],
               sources: pd.DataFrame, info: Union[list, pd.DataFrame]):
        """Extend the current SampleSet with the given data.

        Always verify that the data was appended in the way you expect based on what input type
        you are preferring to work with.

        Args:
            spectra: spectra to append to the current spectra DataFrame
            sources: sources to append to the current sources DataFrame
            info: info to append to the current info DataFrame

        Raises:
            `ValueError` when internal `DataFrame` lengths do not match
        """
        if not spectra.shape[0] == sources.shape[0] == info.shape[0]:
            msg = "The number of rows in each of the required positional arguments must be same."
            raise ValueError(msg)

        self._spectra = self._spectra\
            .append(pd.DataFrame(spectra), ignore_index=True, sort=True)\
            .fillna(0)

        new_sources_column_index = pd.MultiIndex.from_tuples(
            sources.keys(),
            names=SampleSet.SOURCES_MULTI_INDEX_NAMES
        )
        new_sources_df = pd.DataFrame(sources.values(), columns=new_sources_column_index)
        self._sources = self._sources\
            .append(new_sources_df, ignore_index=True, sort=True)\
            .fillna(0)

        self._info = self._info\
            .append(pd.DataFrame(info), ignore_index=True, sort=True)

    def get_all_channel_energies(self, fractional_energy_bins=None) -> np.ndarray:
        """Get the energy (in keV) represented by the lower edge of each channel for all samples.

        See docstring for `_channels_to_energies()` for more details.

        Args:
            fractional_energy_bins: array of fractions representing the spacing of
                the channels (for arbitrary channel structures)

        Returns:
            2-D array of energy values in keV for all samples
        """
        if fractional_energy_bins is None:
            fractional_energy_bins = np.linspace(0, 1, self.n_channels)

        all_channel_energies = []
        for i in range(self.n_samples):
            channel_energies = self.get_channel_energies(i, fractional_energy_bins)
            all_channel_energies.append(channel_energies)

        return all_channel_energies

    def get_channel_energies(self, sample_index, fractional_energy_bins=None) -> np.ndarray:
        """Get the energy (in keV) represented by the lower edge of each channel for a
        single sample at a specific index.

        See docstring for `_channels_to_energies()` for more details.

        Args:
            sample_index: index of the specific sample for which to obtain energies
            fractional_energy_bins: array of fractions representing the spacing of
                the channels (for arbitrary channel structures)

        Returns:
            Array of energy values in keV
        """
        if fractional_energy_bins is None:
            fractional_energy_bins = np.linspace(0, 1, self.n_channels)

        # TODO: raise error if ecal info is missing for any row

        offset, gain, quadratic, cubic, low_energy = self.ecal[sample_index]
        channel_energies = self._channels_to_energies(
            fractional_energy_bins,
            offset, gain, quadratic, cubic, low_energy
        )

        return channel_energies

    def get_confidences(self, fg_seeds_ss: SampleSet, bg_seed_ss: SampleSet = None,
                        bg_cps: float = None, is_lpe: bool = False,
                        confidence_func: Callable = None,
                        **confidence_func_kwargs) -> np.ndarray:
        """Get confidence measure for predictions as a NumPy array.

        Args:
            fg_seeds_ss: `Sampleset` containing foreground seeds
            bg_seed_ss: `Sampleset` containing a single background seed
            bg_cps: count rate used to scale `bg_seed_ss`
            is_lpe: whether predictions are for multi-class classification or label proportion
                estimation (LPE)
            confidence_func: metric function used to compare spectral reconstructions and
            confidence_func_kwargs: kwargs for confidence metric function

        Returns:
            Array containing confidence metric for each prediction
        """
        if fg_seeds_ss.spectra_type != SpectraType.Foreground:
            msg = (
                "`fg_seeds_ss` must have a `spectra_type` of `Foreground`. "
                f"Its `spectra_type` is `{fg_seeds_ss.spectra_type}`."
            )
            raise ValueError(msg)
        if self.spectra_type == SpectraType.Gross:
            if self.info.total_counts.isna().any() or (self.info.total_counts <= 0).any():
                msg = (
                    "This `SampleSet` must have the `info.total_counts` column filled in "
                    "with values greater than zero."
                    "If there are samples with spectra of all zeros, remove them first."
                )
                raise ValueError(msg)
            if self.info.live_time.isna().any() or (self.info.live_time <= 0).any():
                msg = (
                    "This `SampleSet` must have the `info.live_time` column filled in "
                    "with values greater than zero."
                    "If there are samples with live of zero, remove them first."
                )
                raise ValueError(msg)
            if not bg_seed_ss:
                raise ValueError("`bg_seed_ss` is required for current `spectra_type` of `Gross`.")
            if bg_seed_ss.n_samples != 1:
                raise ValueError("`bg_seed_ss` must have exactly one sample.")
            if bg_seed_ss.spectra_type != SpectraType.Background:
                msg = (
                    "`bg_seed_ss` must have a `spectra_type` of `Background`. "
                    f"Its `spectra_type` is `{bg_seed_ss.spectra_type}`."
                )
                raise ValueError(msg)
            if not bg_cps or bg_cps <= 0:
                raise ValueError("Positive background count rate required.")

        if confidence_func is None:
            from riid.losses import jensen_shannon_divergence
            confidence_func = jensen_shannon_divergence

        if is_lpe:
            recon_proportions = self.prediction_probas.values
        else:
            max_indices = np.argmax(self.prediction_probas.values, axis=1)
            recon_proportions = np.zeros(self.prediction_probas.values.shape)
            recon_proportions[np.arange(self.n_samples), max_indices] = 1

        reconstructions = np.dot(recon_proportions, fg_seeds_ss.spectra.values)

        if self.spectra_type == SpectraType.Gross:
            bg_spectrum = bg_seed_ss.spectra.iloc[0].values
            normalized_bg_spectrum = bg_spectrum / bg_spectrum.sum()

            bg_counts = bg_cps * self.info.live_time.values
            fg_counts = (self.info.total_counts.values - bg_counts).clip(1)

            scaled_bg_spectra = get_expected_spectra(normalized_bg_spectrum, bg_counts)
            scaled_fg_spectra = reconstructions * fg_counts[:, np.newaxis]

            reconstructions = scaled_fg_spectra + scaled_bg_spectra

        confidences = confidence_func(
            reconstructions.astype(np.float64),
            self.spectra.values,
            **confidence_func_kwargs
        )
        confidences = np.array(confidences)
        return confidences

    def _get_spectral_distances(self, distance_func=distance.jensenshannon) -> np.array:
        n_samples = self.n_samples
        spectra = self.spectra.values.copy()
        spectrum_pairs = [
            (spectra[i], spectra[j])
            for i in range(n_samples)
            for j in range(i, n_samples)
        ]
        left_spectrum, right_spectrum = zip(*spectrum_pairs)
        distance_values = distance_func(left_spectrum, right_spectrum, axis=1)

        return distance_values

    def get_multiclass_jsds(self, fg_seeds_ss: SampleSet, target_level: str) -> list:
        """For each sample, this constructs a dictionary containing seed name to JSD using
        the sample's top prediction.

        Args:
            fg_seeds_ss: `SampleSet` from which to pull seeds for computing JSD
            prediction_target_level: the level at which predictions were made by a model

        Returns:
            List of dictionaries where keys are seed name and values are JSD
        """
        test_prediction_labels = self.get_predictions(target_level)
        fg_seed_labels = fg_seeds_ss.get_labels(target_level)

        jsds = []
        for i, pred_label in enumerate(test_prediction_labels):
            test_spectrum = self.spectra.iloc[i]
            seeds_for_pred = fg_seeds_ss[fg_seed_labels == pred_label]
            seed_labels_for_pred = seeds_for_pred.get_labels("Seed")
            jsds_for_sample = {}
            for j, seed_label in enumerate(seed_labels_for_pred):
                seed_spectrum = seeds_for_pred.spectra.iloc[j]
                jsd = distance.jensenshannon(test_spectrum, seed_spectrum)
                jsds_for_sample[seed_label] = jsd
            jsds.append(jsds_for_sample)
        return jsds

    def get_spectral_distance_matrix(self, distance_func=distance.jensenshannon,
                                     target_level="Seed") -> pd.DataFrame:
        """Compute the distance between all pairs of spectra.

        This method is intended for use on seed spectra (i.e., templates).
        Calling this method on large samplesets, such as those produced by the static
        synthesizer, is not recommended.
        This method only computes the upper triangle and diagonal of the matrix since the lower
        triangle would be a copy of the upper triangle.
        The diagonal is computed, for when Poisson noise is present, for two main reasons:

        - the distance between the same source will effectively never be zero.
        - there is sometimes research interest in comparing the diagonal to the upper triangle.

        Args:
            distance_func: specific distance function to use
            target_level: `SampleSet.sources` column level to use

        Returns:
            A 2-D array of distance values
        """
        distance_values = self._get_spectral_distances(distance_func)
        row_labels = self.get_labels(target_level=target_level)
        col_labels = self.sources.columns.get_level_values(level=target_level)
        distance_df = _get_distance_df_from_values(distance_values, row_labels, col_labels)

        return distance_df

    def get_labels(self, target_level="Isotope", max_only=True,
                   include_value=False, min_value=0.01,
                   level_aggregation=None):
        """Get row ground truth labels for each sample based on `sources` values.

        See docstring for `_get_row_labels()` for more details.
        """
        labels = _get_row_labels(
            self.sources,
            target_level=target_level,
            max_only=max_only,
            include_value=include_value,
            min_value=min_value,
            level_aggregation=level_aggregation,
        )
        return labels

    def get_predictions(self, target_level="Isotope", max_only=True,
                        include_value=False, min_value=0.01,
                        level_aggregation=None):
        """Get row predictions for each spectrum based on `prediction_probas` values.

        See docstring for `_get_row_labels()` for more details.
        """
        labels = _get_row_labels(
            self.prediction_probas,
            target_level=target_level,
            max_only=max_only,
            include_value=include_value,
            min_value=min_value,
            level_aggregation=level_aggregation,
        )
        return labels

    def get_samples(self) -> np.ndarray:
        """Get `spectra` as a NumPy array.

        Replaces NaN values with 0.

        Returns:
            Array containing spectrum data
        """
        spectra_values = np.nan_to_num(self._spectra)
        return spectra_values

    def get_source_contributions(self, target_level="Isotope") -> np.ndarray:
        """Get `sources` as a NumPy array.

        Replaces NaN values with 0.

        Returns:
            Array containing the ground truth contributions for each sample
        """
        collapsed_sources = self.sources.T.groupby(target_level).sum().T
        sources_values = np.nan_to_num(collapsed_sources)
        return sources_values

    def normalize(self, p: float = 1, clip_negatives: bool = True):
        """Apply L-p normalization to `spectra` in place.

        Default is L1 norm (p=1) can be interpreted as a forming a probability mass function (PMF).
        L2 norm (p=2) is unit energy (sum of squares == 1).

        Reference: https://en.wikipedia.org/wiki/Parseval's_theorem

        Args:
            p: exponent to which the spectra values are raised
            clip_negatives: whether negative values will be removed from the spectra and
                replaced with 0
        """
        if clip_negatives:
            self.clip_negatives()
        elif (self._spectra.values < 0).sum():
            logging.warning(
                "You are performing a normalization operation on spectra which contain negative " +
                "values.  Consider applying `clip_negatives`."
            )

        energy = (self._spectra.values ** p).sum(axis=1)[:, np.newaxis]
        energy[energy == 0] = 1
        self._spectra = pd.DataFrame(data=self._spectra.values / energy ** (1/p))
        if p == 1:
            self.spectra_state = SpectraState.L1Normalized
        elif p == 2:
            self.spectra_state = SpectraState.L2Normalized
        else:
            self.spectra_state = SpectraState.Unknown

    def normalize_sources(self):
        """Normalize `sources` such that rows sum to 1."""
        row_sums = self._sources.sum(axis=1)
        row_sums[row_sums == 0] = 1
        self._sources = self._sources.clip(0).divide(
            row_sums,
            axis=0,
        )

    def replace_nan(self, replace_value: float = 0):
        """Replace np.nan values.

        Args:
            replace_value: value with which NaN will be replaced

        """
        self._spectra.replace(np.nan, replace_value)
        self._sources.replace(np.nan, replace_value)
        self._info.replace(np.nan, replace_value)

    def sample(self, n_samples: int, random_seed: int = None) -> SampleSet:
        """Randomly sample the SampleSet.

        Args:
            n_samples: number of random samples to be returned
            random_seed: seed value for random number generation

        Returns:
            `Sampleset` of randomly selected measurements
        """
        if random_seed:
            random.seed(random_seed)

        if n_samples > self.n_samples:
            n_samples = self.n_samples

        random_indices = random.sample(self.spectra.index.values.tolist(), n_samples)
        random_mask = np.isin(self.spectra.index, random_indices)
        return self[random_mask]

    def set_dead_time_proportions(self):
        """Computes dead time proportion for each sample and saves it in the `SampleSet` info.
        """
        dead_time_props = self._get_dead_time_proportions()
        self.info[self.DEAD_TIME_PROP_INFO_KEY] = dead_time_props

    def shuffle(self, inplace: bool = True, random_state: int = None) -> SampleSet:
        """Randomly reorder all DataFrames.

        Args:
            inplace: whether to perform the operation in-place
            random_state: random seed for reproducing specific shuffles

        Returns:
            `SampleSet` object when `inplace` is False
        """
        new_row_ordering = np.arange(self.n_samples)
        np.random.default_rng(random_state).shuffle(new_row_ordering)

        new_ss = self if inplace else self[:]

        new_ss.spectra = self.spectra.iloc[new_row_ordering]\
            .reset_index(drop=True)
        new_ss.sources = self.sources.iloc[new_row_ordering]\
            .reset_index(drop=True)
        new_ss.info = self.info.iloc[new_row_ordering]\
            .reset_index(drop=True)
        if not new_ss.prediction_probas.empty:
            new_ss.prediction_probas = self.prediction_probas.iloc[new_row_ordering]\
                .reset_index(drop=True)

        if not inplace:
            return new_ss

    def sources_columns_to_dict(self, target_level="Isotope") -> Union[dict, list]:
        """Convert the MultiIndex columns of `sources` to a dictionary.

        Note: depending on `target_level` and `sources` columns, duplicate values are possible.

        Args:
            target_level: `SampleSet.sources` column level to use

        Returns:
            If `target_level` is "Category" or "Isotope," then a dict is returned.
            If `target_level` is "Seed," then a list is returned.

        Raises:
            `ValueError` when `target_level` is invalid
        """
        self._check_target_level(
            target_level,
            levels_allowed=SampleSet.SOURCES_MULTI_INDEX_NAMES
        )

        d = {}
        column_tuples = self.sources.columns.to_list()
        if target_level == "Seed":
            d = column_tuples
        elif target_level == "Isotope":
            for t in column_tuples:
                _, i, _ = t
                if i not in d:
                    d[i] = [t]
                if t not in d[i]:
                    d[i].append(t)
        else:  # target_level == "Category":
            for t in column_tuples:
                c, i, _ = t
                if c not in d:
                    d[c] = {}
                if i not in d[c]:
                    d[c][i] = [t]
                if t not in d[c][i]:
                    d[c][i].append(t)

        return d

    def split_fg_and_bg(self, bg_seed_names: Iterable = DEFAULT_BG_SEED_NAMES) \
            -> Tuple[SampleSet, SampleSet]:
        """Split the current `SampleSet` into two new `SampleSet`s, one containing only foreground
        sources and the other containing only background sources.

        Foreground sources are assumed to be anything that is not designated as a background source.

        Args:
            bg_seeds_names: names of the seeds which are considered background sources.
                This list be customized to also extract atypical background sources such as
                calibration sources.

        Returns:
            Two `SampleSet`s, the first containing only foregrounds and the second only
            containing backgrounds.
        """
        seed_labels = self.get_labels(target_level="Seed")
        bg_mask = seed_labels.isin(bg_seed_names)

        fg_seeds_ss = self[~bg_mask]
        fg_seeds_ss.drop_sources_columns_with_all_zeros()
        fg_seeds_ss.spectra_type = SpectraType.Foreground
        fg_seeds_ss.spectra_state = self.spectra_state

        bg_seeds_ss = self[bg_mask]
        bg_seeds_ss.drop_sources_columns_with_all_zeros()
        bg_seeds_ss.spectra_type = SpectraType.Background
        bg_seeds_ss.spectra_state = self.spectra_state

        return fg_seeds_ss, bg_seeds_ss

    def squash(self) -> SampleSet:
        """Combine all rows of the `SampleSet` into a single row.

        All data suited for summing is summed, otherwise the info from the first
        sample is used.

        Returns:
            A flattened `SampleSet`
        """
        flat_spectra = self.spectra.sum(axis=0).to_frame().T
        flat_sources = self.sources.sum(axis=0).to_frame().T
        flat_prediction_probas = self.prediction_probas.sum(axis=0).to_frame().T
        flat_info = self.info.iloc[0].to_frame().T

        if "description" in flat_info:
            flat_info.description = "squashed"
        if "live_time" in flat_info:
            flat_info.live_time = self.info.live_time.sum()
        if "real_time" in flat_info:
            flat_info.real_time = self.info.real_time.sum()
        if "total_counts" in flat_info:
            flat_info.total_counts = self.info.total_counts.sum()
        if "snr" in flat_info:
            flat_info.snr = self.info.snr.sum()

        # Create a new SampleSet with the flattened data
        flat_ss = SampleSet()
        flat_ss.spectra = flat_spectra
        flat_ss.sources = flat_sources
        flat_ss.prediction_probas = flat_prediction_probas
        flat_ss.info = flat_info
        # Copy dictionary and other non-DataFrame objects from full ss
        flat_ss._detector_info = self._detector_info
        flat_ss._synthesis_info = self._synthesis_info
        flat_ss._measured_or_synthetic = self._measured_or_synthetic

        return flat_ss

    def to_hdf(self, path: str | Path, verbose=False, **kwargs):
        """Write the `SampleSet` to disk as a HDF file.

        Args:
            path: file path at which to save as an HDF file
            verbose: whether to display detailed output
            kwargs: additional arguments to be passed to the `Pandas.HDFStore` constructor

        Raises:
            `ValueError` when provided path extension is invalid
        """
        path = Path(path)
        if path.suffix != riid.SAMPLESET_HDF_FILE_EXTENSION:
            logging.warning(f"Path does not end in {riid.SAMPLESET_HDF_FILE_EXTENSION}")

        _write_hdf(self, path, **kwargs)
        if verbose:
            logging.info(f"Saved SampleSet to '{path}'")

    def to_json(self, path: str | Path, verbose=False):
        """Write the `SampleSet` to disk as a JSON file.

        Warning: it is not recommended that you use this on large `SampleSet` objects.
        Consider `SampleSet.to_hdf()` instead in such cases.

        Args:
            path: file path at which to save as an HDF file
            verbose: whether to display detailed output

        Raises:
            `ValueError` when provided path extension is invalid
        """
        path = Path(path)
        if path.suffix != riid.SAMPLESET_JSON_FILE_EXTENSION:
            logging.warning(f"Path does not end in {riid.SAMPLESET_JSON_FILE_EXTENSION}")

        _write_json(self, path)
        if verbose:
            logging.info(f"Saved SampleSet to '{path}'")

    def to_pcf(self, path: str | Path, verbose=False):
        """Write the `SampleSet` to disk as a PCF.

        Args:
            path: file path at which to save as a PCF
            verbose: whether to display detailed output

        Raises:
            `ValueError` when provided path extension is invalid
        """
        path = Path(path)
        if path.suffix != riid.PCF_FILE_EXTENSION:
            logging.warning(f"Path does not end in {riid.PCF_FILE_EXTENSION}")

        _dict_to_pcf(_ss_to_pcf_dict(self, verbose), path, verbose)

        if verbose:
            logging.info(f"Saved SampleSet to '{path}'")

    def update_timestamp(self):
        """Set the timestamp for all samples to the current UTC date and time."""
        self.info.timestamp = _get_utc_timestamp()

    def upsample_spectra(self, target_bins: int = 4096):
        """Uniformly up-bin spectra.

        Args:
            target_bins: number of bins into which the current spectra
                should be split.
        """
        if target_bins % self.n_channels == 0:
            n_per_group = int(target_bins / self.n_channels)
            transformation = np.zeros([target_bins, self.n_channels])
            for i in range(self.n_channels):
                transformation[(i * n_per_group):((i * n_per_group) + n_per_group), i] = \
                                    1.0 / n_per_group
        else:
            raise ValueError(
                ("Desired number of bins ({}) is cannot"
                 " be uniformly mapped from existing"
                 " channels ({})").format(target_bins, self.n_channels))
        self._spectra = pd.DataFrame(
            data=np.matmul(
                self._spectra.values,
                transformation.T))


def read_hdf(path: str | Path) -> SampleSet:
    """Read an HDF file in as a `SampleSet` object.

    Args:
        path: path for the HDF file to be read in

    Returns:
        `SampleSet` object
    """
    expanded_path = Path(path).expanduser()
    if not expanded_path.is_file():
        raise FileNotFoundError(f"No file found at location '{expanded_path}'.")

    ss = _read_hdf(expanded_path)

    if not ss:
        raise FileNotFoundError(f"No data found at location '{expanded_path}'.")

    return ss


def read_pcf(path: str | Path, verbose: bool = False) -> SampleSet:
    """Read a PCF file in as a `SampleSet` object.

    Args:
        path: path for the PCF file to be read in
        verbose: whether to show verbose function output in terminal

    Returns:
        `Sampleset` object
    """
    expanded_path = Path(path).expanduser()
    return _pcf_dict_to_ss(_pcf_to_dict(expanded_path, verbose), verbose)


def _dict_to_bulleted_list(data_dict: dict, level=0, indent=4, bullet="-") -> str:
    lines = []
    level_key_width = max([len(x) for x in data_dict.keys()])
    for k, v in sorted(data_dict.items()):
        key = f"{k}:".ljust(level_key_width + 1)
        line = f"{' ' * (indent * level)}{bullet} {key}"
        if type(v) is dict:
            lines.append(line)
            sub_lines = _dict_to_bulleted_list(v, level + 1, indent, bullet)
            lines.extend(sub_lines)
        else:
            line += f" {v}"
            lines.append(line)
    return lines


def _get_utc_timestamp():
    now_utc = datetime.now(timezone.utc)
    ts = now_utc.isoformat(sep=" ", timespec="seconds")
    return ts


def _get_row_labels(df: pd.DataFrame, target_level: str = "Isotope", max_only: bool = True,
                    include_value: bool = False, min_value: float = 0.01,
                    level_aggregation: str = "sum") -> pd.Series:
    """Interprets the cell values in conjunction with the columns to determine an
    appropriate label for each row.

    Args:
        df: `DataFrame` with `MultiIndex` columns and named levels
        target_level: level of the columns to use for row labels
        max_only: for each row, whether to only use the column
            containing the max value.  When False, all column names
            containing a cell value >= `min_value` will be obtained.
        include_value: whether to include the cell value
            alongside the column name(s).
        min_value: when `max_only=False`, this filters out columns
            containing values less than this value. Note: this applies
            to values AFTER aggregation occurs.
        level_aggregation: method to use for combining
            cell values which have the same column name at the
            `target_level`.  Acceptable values are "sum" or "mean",
            otherwise this is ignored.

    Returns:
        Pandas `Series` where each entry is the row label
    """
    if max_only:
        if level_aggregation == "sum":
            values = df.T.groupby(target_level).sum().T
            labels = values.idxmax(axis=1)
        elif level_aggregation == "mean":
            values = df.T.groupby(target_level).mean().T
            labels = values.idxmax(axis=1)
        else:
            levels_to_drop = [
                x for x in SampleSet.SOURCES_MULTI_INDEX_NAMES
                if x != target_level and x in df.columns.names
            ]
            values = df.droplevel(levels_to_drop, axis=1)
            labels = values.idxmax(axis=1)
        if include_value:
            values = values.max(axis=1)
            labels = [f"{x} ({y:.2f})" for x, y in zip(labels, values)]
    else:  # Much slower
        if level_aggregation == "sum":
            values = df.T.groupby(target_level).sum().T
        elif level_aggregation == "mean":
            values = df.T.groupby(target_level).mean().T
        else:
            values = df
        mask = values.ge(min_value).values
        cols = values.columns.get_level_values(target_level).values
        if include_value:
            labels = [
                " + ".join([
                    f"{c} ({v:.2f})"
                    for c, v in zip(cols[x], values.iloc[i][x])
                ])
                for i, x in enumerate(mask)
            ]
        else:
            labels = [" + ".join(cols[x]) for x in mask]

    return pd.Series(labels)


def _validate_hdf_store_keys(keys: list):
    """Validates the `SampleSet` keys based on whether it contains the minimum required
    set of keys.

    Args:
        keys: collection of keys from the dataset being converted to a `SampleSet`

    Raises:
        `InvalidSampleSetFileError` when the set of keys provided does not include a required key
    """
    required_keys = [
        "/info",
        "/spectra",
        "/sources",
        "/prediction_probas"
    ]
    for rk in required_keys:
        if isinstance(rk, tuple):
            if all([ork not in keys for ork in rk]):
                msg = f"SampleSet did not contain at least one of the following pieces \
                        of required data: {rk}"
                raise InvalidSampleSetFileError(msg)
        else:
            if rk not in keys:
                msg = f"SampleSet did not contain the following piece of required data: {rk}"
                raise InvalidSampleSetFileError()


def _read_hdf(path: Path) -> SampleSet:
    """Read `SampleSet` from an HDF file.

    Args:
        path: path where HDF file is to be read from

    Returns:
        `SampleSet` object
    """
    store = pd.HDFStore(path, mode="r")
    store_keys = store.keys()
    _validate_hdf_store_keys(store_keys)

    # Pull data from data store
    spectra = store.get("spectra")
    info = store.get("info")
    sources = store.get("sources")
    prediction_probas = store.get("prediction_probas")
    other_info = store.get("other_info")
    store.close()

    # Build SampleSet object
    ss = SampleSet()
    ss.spectra = spectra
    ss.sources = sources
    ss.info = info
    ss.prediction_probas = prediction_probas
    if "spectra_state" in other_info:
        ss.spectra_state = SpectraState(other_info["spectra_state"].iloc[0])
    if "spectra_type" in other_info:
        ss.spectra_type = SpectraType(other_info["spectra_type"].iloc[0])
    if "measured_or_synthetic" in other_info:
        ss.measured_or_synthetic = other_info["measured_or_synthetic"].iloc[0]
    if "detector_info" in other_info:
        ss.detector_info = other_info["detector_info"].iloc[0]
    if "synthesis_info" in other_info:
        ss.synthesis_info = other_info["synthesis_info"].iloc[0]
    if "classified_by" in other_info:
        ss.classified_by = other_info["classified_by"].iloc[0]

    return ss


def _write_hdf(ss: SampleSet, output_path: Path, **kwargs):
    """Write a `SampleSet` to an HDF file.

    Args:
        ss: `SampleSet` object to be written out
        output_path: path where file is to be written
        kwargs: additional arguments to be passed to the `Pandas.HDFStore` constructor
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        store_kwargs = {
            "complevel": 3,
            "complib": "blosc:blosclz",
        }
        store_kwargs.update(kwargs)

        put_kwargs = {
            "format": "fixed",
            "track_times": False,
        }
        with pd.HDFStore(output_path, "w", **store_kwargs) as store:
            store.put("spectra", ss.spectra, **put_kwargs)
            store.put("sources", ss.sources, **put_kwargs)
            store.put("info", ss.info, **put_kwargs)
            store.put("prediction_probas", ss.prediction_probas, **put_kwargs)

            other_info = {
                "spectra_state": ss.spectra_state,
                "spectra_type": ss.spectra_type,
                "measured_or_synthetic": ss.measured_or_synthetic,
                "detector_info": ss.detector_info,
                "synthesis_info": ss.synthesis_info,
                "classified_by": ss.classified_by,
            }
            store.put("other_info", pd.DataFrame(data=[other_info]), **put_kwargs)


def _ss_to_pcf_dict(ss: SampleSet, verbose=False) -> dict:
    """Convert a `SampleSet` to a dictionary representation.

    Args:
        ss: `SampleSet` object to convert to a dictionary
        verbose: whether to show detailed output

    Returns:
        Dictionary containing the values from the Sampleset
    """
    n_channels = int(ss.n_channels)
    n_records_per_spectrum = int((n_channels / 64) + 1)

    if "pcf_metadata" in ss.detector_info and ss.detector_info["pcf_metadata"]:
        pcf_header = ss.detector_info["pcf_metadata"]
    else:
        pcf_header = {
            "NRPS": n_records_per_spectrum,
            "Version": "DHS",
            "last_mod_hash": " " * 7,
            "UUID": " " * 36,
            "Inspection": " " * 16,
            "Lane_Number": 0,
            "Measurement_Remark": " " * 26,
            "Intrument_Type": " " * 28,
            "Manufacturer": " " * 28,
            "Instrument_Model": " " * 18,
            "Instrument_ID": " " * 18,
            "Item_Description": " " * 20,
            "Item_Location": " " * 16,
            "Measurement_Coordinates": " " * 16,
            "Item_to_detector_distance": 0,
            "Occupancy_Number": 0,
            "Cargo_Type": " " * 16,
            "SRSI": 83,
            "DevType": "DeviationPairsInFile",
        }

    spectra = []

    isotopes, seeds = pd.Series([], dtype=pd.StringDtype()), pd.Series([], dtype=pd.StringDtype())
    if not ss.sources.empty:
        isotope_level_name = SampleSet.SOURCES_MULTI_INDEX_NAMES[1]
        seed_level_names = SampleSet.SOURCES_MULTI_INDEX_NAMES[2]
        if isotope_level_name in ss.sources.columns.names:
            isotopes = ss.get_labels(target_level=isotope_level_name)
        if seed_level_names in ss.sources.columns.names:
            seeds = ss.get_labels(target_level=seed_level_names)

    sample_range = range(ss.n_samples)
    if verbose:
        sample_range = tqdm.tqdm(
            sample_range,
            desc="Converting sample set"
        )

    info_timestamps = ss.info.timestamp.fillna("")
    info_tags = ss.info.tag.fillna("")
    info_zero_fill = ss.info.infer_objects(copy=False).astype(float, errors="ignore")
    for i in sample_range:
        title = isotopes[i] if not isotopes.empty else NO_ISOTOPE
        description = ss.info.description.fillna("").iloc[i]
        source = seeds[i] if not seeds.empty else NO_SEED
        compressed_text_buffer = _pack_compressed_text_buffer(title, description, source)

        header = {
            "Compressed_Text_Buffer": compressed_text_buffer,
            "Energy_Calibration_Offset": info_zero_fill.ecal_order_0.iloc[i],
            "Energy_Calibration_Gain": info_zero_fill.ecal_order_1.iloc[i],
            "Energy_Calibration_Quadratic": info_zero_fill.ecal_order_2.iloc[i],
            "Energy_Calibration_Cubic": info_zero_fill.ecal_order_3.iloc[i],
            "Energy_Calibration_Low_Energy": info_zero_fill.ecal_low_e.iloc[i],
            "Live_Time": info_zero_fill.live_time.iloc[i],
            "Total_time_per_real_time": info_zero_fill.real_time.iloc[i],
            "Number_of_Channels": n_channels,
            "Date-time_VAX": info_timestamps.iloc[i],
            "Occupancy_Flag": info_zero_fill.occupancy_flag.iloc[i],
            "Tag": info_tags.iloc[i],
            "Total_Neutron_Counts": info_zero_fill.neutron_counts.iloc[i],
        }

        spectrum = ss.spectra.iloc[i, :].values.astype(float).tolist()
        spectra.append({"header": header, "spectrum": spectrum})

    return {"header": pcf_header, "spectra": spectra}


def _pcf_dict_to_ss(pcf_dict: dict, verbose=True):
    """Convert a PCF dictionary into a `SampleSet`.

    Args:
        pcf_dict: dictionary representation of a PCF
        verbose: whether to display detailed output

    Returns:
        `Sampleset` object containing the PCF dict values
    """
    if not pcf_dict["spectra"]:
        return

    num_spectra = len(pcf_dict["spectra"])
    num_channels = len(pcf_dict["spectra"][0]["spectrum"])

    sources = []
    infos = []
    spectra = np.ndarray((num_spectra, num_channels))

    for i in range(0, num_spectra):
        spectrum = pcf_dict["spectra"][i]
        compressed_text_buffer = spectrum["header"]["Compressed_Text_Buffer"]

        title, description, source = _unpack_compressed_text_buffer(compressed_text_buffer)

        if not title and not source:
            source = NO_ISOTOPE
            title = NO_ISOTOPE
            category = NO_CATEGORY
        else:
            if not title:
                title = _find_isotope(source, verbose)
            elif not source:
                source = title
                title = _find_isotope(source, verbose)
            category = _find_category(title)

        # Extract any additional information from the source string
        distance_search = re.search("@ ([0-9,.]+)", source)
        if distance_search:
            distance = float(distance_search.group(1)) / 100
        else:
            distance = np.nan
        an_finds = re.findall("an=([0-9]+)", source)
        if an_finds:
            an = int(an_finds[0])
        else:
            an = None
        ad_finds = re.findall("ad=([0-9, .]+)", source)
        if ad_finds:
            ad_string = ad_finds[0].replace(",", "")
            ad = float(ad_string)
        else:
            ad = None

        order_0 = float(spectrum["header"]["Energy_Calibration_Offset"])
        order_1 = float(spectrum["header"]["Energy_Calibration_Gain"])
        order_2 = float(spectrum["header"]["Energy_Calibration_Quadratic"])
        order_3 = float(spectrum["header"]["Energy_Calibration_Cubic"])
        low_E = float(spectrum["header"]["Energy_Calibration_Low_Energy"])

        info = {
            "description": description,
            "timestamp": spectrum["header"]["Date-time_VAX"],
            "live_time": spectrum["header"]["Live_Time"],
            "real_time": spectrum["header"]["Total_time_per_real_time"],
            "total_counts": sum(spectrum["spectrum"]),
            "neutron_counts": spectrum["header"]["Total_Neutron_Counts"],
            "distance_cm": distance,
            "ecal_order_0": order_0,
            "ecal_order_1": order_1,
            "ecal_order_2": order_2,
            "ecal_order_3": order_3,
            "ecal_low_e": low_E,
            "atomic_number": an,
            "areal_density": ad,
            "occupancy_flag": spectrum["header"]["Occupancy_Flag"],
            "tag": spectrum["header"]["Tag"],
        }

        infos.append(info)
        sources.append((category, title, source))
        spectra[i, :] = np.array(spectrum["spectrum"])

    n_sources = len(sources)
    n_samples = len(spectra)
    new_index = pd.MultiIndex.from_tuples(sources, names=SampleSet.SOURCES_MULTI_INDEX_NAMES)
    sources_df = pd.DataFrame(np.zeros((n_samples, n_sources)), columns=new_index)
    for i, key in enumerate(sources):
        sources_df.loc[i, key] = 1.0

    ss = SampleSet()
    ss.spectra = pd.DataFrame(data=spectra)
    ss.sources = sources_df
    ss.info = pd.DataFrame(data=infos)
    ss.measured_or_synthetic = "synthetic",
    ss.detector_info["pcf_metadata"] = pcf_dict["header"]

    return ss


def _write_json(ss: SampleSet, output_path: Path):
    ss_dict = _ss_to_pcf_dict(ss)
    ss_dict["detector_info"] = ss.detector_info
    with open(output_path, "w") as fout:
        json.dump(ss_dict, fout, indent=4)


def read_json(path: str | Path) -> SampleSet:
    expanded_path = Path(path).expanduser()
    with open(expanded_path, "r") as fin:
        ss_dict = json.load(fin)
    ss = _pcf_dict_to_ss(ss_dict)
    ss.detector_info = ss_dict["detector_info"]
    return ss


def _get_distance_df_from_values(distance_values: np.ndarray,
                                 row_labels: pd.Series, col_labels: pd.Series) -> pd.DataFrame:
    n_rows = row_labels.shape[0]
    n_cols = col_labels.shape[0]
    distance_df = pd.DataFrame(
        np.zeros((n_rows, n_cols)),
        index=row_labels,
        columns=col_labels,
    )
    n_values = len(distance_values)
    label_pairs = [
        (row_labels[i], col_labels[j])
        for i in range(n_rows)
        for j in range(i, n_cols)
    ]

    for i in range(n_values):
        value = distance_values[i]
        r, c = label_pairs[i]
        distance_df.at[r, c] = value

    return distance_df


def _get_energy_roi_masks(rois: list, energies: np.ndarray) -> np.ndarray:
    masks = np.zeros(energies.shape, dtype=bool)
    for (elow, ehigh) in rois:
        roi_mask = (elow <= energies) & (energies < ehigh)
        masks |= roi_mask
    return masks


class InvalidSampleSetFileError(Exception):
    """Missing or invalid keys in a file being read in as a `SampleSet`."""
    pass


class InvalidSampleCountError(Exception):
    """Incorrect number of samples encountered for the operation."""
    pass


class SpectraStateMismatchError(Exception):
    """Two `SampleSet`s have different `spectra_state` values."""
    pass


class ChannelCountMismatchError(Exception):
    """Two `SampleSet`s have different `n_channels` values."""
    pass
