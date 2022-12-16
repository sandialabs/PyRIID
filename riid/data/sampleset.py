# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains the SampleSet class and other SampleSet-related functions."""
from __future__ import \
    annotations  # Enables SampleSet hints inside SampleSet itself

import copy
import logging
import os
import random
import re
import warnings
from datetime import datetime
from enum import Enum
from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d

import riid
from riid.data.labeling import (NO_CATEGORY, NO_ISOTOPE, NO_SEED,
                                _find_category, _find_isotope)
from riid.gadras.pcf import (_dict_to_pcf, _pack_compressed_text_buffer,
                             _pcf_to_dict, _unpack_compressed_text_buffer)


class SpectraState(Enum):
    """Enumerates the potential states of spectra within a SampleSet."""
    Unknown = 0
    Counts = 1
    L1Normalized = 2
    L2Normalized = 3


class SampleSet():
    """Container for collection of samples."""
    # pylint: disable=R0902
    # pylint: disable=R0904
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
        "bg_counts",
        "bg_counts_expected",
        "fg_counts",
        "fg_counts_expected",
        "gross_counts",
        "gross_counts_expected",
        "neutron_counts",
        "snr",
        "snr_expected",
        "sigma",
        "sigma_expected",
        "distance_cm",
        *ECAL_INFO_COLUMNS,
        "areal_density",
        "atomic_number",
        "occupancy_flag",
        "tag",
    )
    DEFAULT_BG_SEED_NAMES = (
        "Cosmic",
        "PotassiumInSoil",
        "UraniumInSoil",
        "ThoriumInSoil",
    )

    def __init__(self):
        """Initializes the SampleSet class and provides default values
        for all class attributes.

        Expected sizes for DataFrames:
            self._spectra:  [n_samples, n_channels]
            self._sources:  [n_samples, n_sources]

        """
        self._spectra = pd.DataFrame()
        self._sources = pd.DataFrame()
        self._info = pd.DataFrame(columns=SampleSet.DEFAULT_INFO_COLUMNS)
        self._detector_info = {}
        self._synthesis_info = {}
        self._prediction_probas = pd.DataFrame()
        self._measured_or_synthetic = None
        self.pyriid_version = riid.__version__
        self.spectra_state = SpectraState.Unknown
        self._classified_by = ""

    def __bool__(self):
        return bool(len(self))

    def __getitem__(self, key: Union[slice, int]):
        selection = key
        if isinstance(key, int):
            selection = slice(key, key+1)

        sub_ss = copy.copy(self)
        sub_ss.spectra = sub_ss.spectra[selection].reset_index(drop=True)
        sub_ss.sources = sub_ss.sources[selection].reset_index(drop=True)
        sub_ss.info = sub_ss.info[selection].reset_index(drop=True)
        if not sub_ss.prediction_probas.empty:
            sub_ss.prediction_probas = sub_ss.prediction_probas[selection].reset_index(drop=True)

        return sub_ss

    def __len__(self):
        return self.n_samples

    def __str__(self):
        isotopes_present = \
            ', '.join(np.unique(self.get_labels())) if not self.sources.empty else "Unknown"
        return (
            "SampleSet Summary:\n"
            f"- # of samples:         {self.n_samples}\n"
            f"- Spectrum channels:    {self.n_channels}\n"
            f"- Detector:             {self.detector_info if self.detector_info else 'Unknown'}\n"
            f"- Predictions present?  {'No' if self.prediction_probas.empty else 'Yes'}\n"
            f"- Sources present:      {isotopes_present}\n"
            f"- PyRIID version:       {self.pyriid_version if self.pyriid_version else 'Unknown'}"
        )

    def __repr__(self):
        return self.__str__()

    # region Properties

    @property
    def category_names(self):
        """Gets the names of the categories involved in this SampleSet."""
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
        """Get or set the detector info on which this SampleSet is based.

        TODO: implement as DataFrame
        """
        return self._detector_info

    @detector_info.setter
    def detector_info(self, value):
        self._detector_info = value

    @property
    def difficulty_score(self, mean=10.0, std=3.0) -> float:
        """Computes a metric representing the "difficulty" of the given SampleSet on a
        scale of 0 to 1, where 0 is easiest and 1 is hardest.

        The difficulty of a SampleSet is the mean of the individual sample difficulties.
        Each sample's difficulty is determined by where its signal strength (sigma)
        falls on the survival function (AKA reliability function, or 1 - CDF) for a
        logistic distribution.
        The decision to use this distribution is based on empirical data showing that
        classifier performance increases in a logistic fashion as signal strength increases.

        The primary use of this function is for situations where ground truth is unknown
        and you want to get an idea of how "difficult" it will be for a model
        to make predictions.  Consider the following example:
            Given a SampleSet for which ground truth is unknown and the signal strength for
            each spectrum is low.
            A model considering, say, 100+ isotopes will see this SampleSet as quite
            difficult, whereas a model considering 5 isotopes will, being more specialized,
            see the SampleSet as easier.
            Of course, this makes the assumption that both models were trained to the
            isotope(s) contained in the test SampleSet.

        Based on the previous example, the unfortunate state of this function is that you
        must know how to pick means and standard deviations which properly reflect the:
            - number of target isotopes
            - amount of variation within each isotope (i.e., shielding, scattering, etc.)
            - detector resolution
        A future goal of ours is to provide updates to this function and docstring which make
        choosing the mean and standard deviation easier/automated based on more of our findings.

        Arguments:
            mean: the sigma value representing the mean of the logistic distribution
            std: the standard deviation of the logistic distribution

        Returns:
            The mean of all sigma values passed through a logistic survival function.

        """
        sigmas: np.ndarray = self.info.sigma.clip(1e-6)
        score = float(stats.logistic.sf(sigmas, loc=mean, scale=std).mean())

        return score

    @property
    def ecal(self):
        """Gets or sets the ecal terms in order:
           (ecal_order_0, ecal_order_1, ecal_order_2, ecal_order_3, ecal_low_e)
        """
        ecal_terms = self.info[list(self.ECAL_INFO_COLUMNS)].values
        return ecal_terms

    @ecal.setter
    def ecal(self, value):
        self.info.loc[:, self.ECAL_INFO_COLUMNS[0]] = value[0]
        self.info.loc[:, self.ECAL_INFO_COLUMNS[1]] = value[1]
        self.info.loc[:, self.ECAL_INFO_COLUMNS[2]] = value[2]
        self.info.loc[:, self.ECAL_INFO_COLUMNS[3]] = value[3]
        self.info.loc[:, self.ECAL_INFO_COLUMNS[3]] = value[4]

    @property
    def info(self):
        """Get or set the info DataFrame."""
        return self._info

    @info.setter
    def info(self, value):
        self._info = value

    @property
    def isotope_names(self):
        """Gets the names of the isotopes involved in this SampleSet."""
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
        """Gets the number of channels included in the spectra DataFrame.
        Channels may also be referred to as bins.
        """
        return self._spectra.shape[1]

    @property
    def n_samples(self):
        """Gets the number of samples included in the spectra dataframe,
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
        """Gets the names of the seeds involved in this SampleSet."""
        return self.sources.columns.get_level_values("Seed")

    @property
    def sources(self):
        """Get or set the sources DataFrame for the SampleSet object.
        """
        return self._sources

    @sources.setter
    def sources(self, value):
        self._sources = value.replace(np.nan, 0)

    @property
    def spectra(self):
        """Get or set the spectra DataFrame for the SampleSet object."""
        return self._spectra

    @spectra.setter
    def spectra(self, value):
        self._spectra = value

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

    def all_spectra_sum_to_one(self) -> bool:
        """Checks if all spectra are normalized to sum to one."""
        return np.all(np.isclose(self.spectra.sum(axis=1).values, 1))

    def as_ecal(self, new_offset: float = None, new_gain: float = None,
                new_quadratic: float = None, new_cubic: float = None,
                new_low_energy: float = None) -> SampleSet:
        """Re-bins spectra based on energy by interpolating the current shape from the current
        binning structure to a new one.

        Warning: this operation is likely to add or remove spectral information
        depending on the new energy calibration values used.

        Args:
            new_offset: new gain value, i.e. the 0-th e-cal term
            new_gain: new gain value, i.e. the 1-th e-cal term
            new_quadratic: new quadratic value, i.e. the 2-th e-cal term
            new_cubic: new cubic value, i.e. the 3-th e-cal term
            new_low_energy: new low energy term

        Raises:
            ValueError: raised if no argument values are provided.

        """
        new_args = [new_offset, new_gain, new_quadratic, new_cubic, new_low_energy]
        if all(v is None for v in new_args):
            raise ValueError("At least one argument value must be provided.")

        ecal_cols = list(self.ECAL_INFO_COLUMNS)
        new_ecal = self.info[ecal_cols].copy()
        if new_offset:
            new_ecal.ecal_order_0 = new_offset
        if new_gain:
            new_ecal.ecal_order_1 = new_gain
        if new_quadratic:
            new_ecal.ecal_order_2 = new_quadratic
        if new_cubic:
            new_ecal.ecal_order_3 = new_cubic
        if new_low_energy:
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
        new_ss.info["gross_counts"] = new_ss.spectra.sum(axis=1)
        new_ss.info[ecal_cols] = new_ecal
        return new_ss

    def as_squashed(self) -> SampleSet:
        """Produces a 1-sample SampleSet from the given SampleSet's data.

        All data suited for summing is summed, otherwise the info from the first
        sample is used.

        Returns:
            flat_ss: A flattened SampleSet.

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
        if "snr_target" in flat_info:
            flat_info.snr_target = self.info.snr_target.sum()
        if "snr" in flat_info:
            flat_info.snr = self.info.snr.sum()
        if "sigma" in flat_info:
            flat_info.sigma = self.info.sigma.sum()
        if "bg_counts" in flat_info:
            flat_info.bg_counts = self.info.bg_counts.sum()
        if "fg_counts" in flat_info:
            flat_info.fg_counts = self.info.fg_counts.sum()
        if "bg_counts_expected" in flat_info:
            flat_info.bg_counts_expected = self.info.bg_counts_expected.sum()
        if "fg_counts_expected" in flat_info:
            flat_info.fg_counts_expected = self.info.fg_counts_expected.sum()
        if "gross_counts" in flat_info:
            flat_info.gross_counts = self.info.gross_counts.sum()
        if "gross_counts_expected" in flat_info:
            flat_info.gross_counts_expected = self.info.gross_counts_expected.sum()

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

    def clip_negatives(self, min_value: float = 0):
        """Sets negative values to min_value.

        Args:
            min_value: Determines the minimum value in spectra to which
                smaller values will be clipped.

        """
        self._spectra = pd.DataFrame(data=self._spectra.clip(min_value))

    def concat(self, ss_list: list):
        """Provides a way to vertically combine many SampleSets.

        Combining of non-DataFrame properties (e.g., detector_info, synthesis_info, etc.)
        is NOT performed - it is the responsiblity of the user to ensure proper bookkeeping for
        these properties when concatenating SampleSets.

        This method is most applicable to SampleSets containing measured data from the same
        detector which could not be made as a single SampleSet because of how measurements had to
        be taken (i.e., measurements were taken by distinct processes separated by time).
        Therefore, the user should avoid concatenating SampleSets when:
            - the data is from different detectors
              (note that SampleSets are given to models for prediction, and models should be
              1-to-1 with physical detectors);
            - a mix of synthetic and measured data would occur
              (this could make an existing `synthesis_info` value ambiguous, but one way to handle
              this would be to add a new column to `info` distinguishing between synthetic and
              measured on a per-sample basis).

        Args:
            ss_list: Defines the list of SampleSets to concatenate.

        Return:
            A single SampleSet object of the concatenated SampleSet data.

        """
        if not ss_list:
            return
        if not isinstance(ss_list, list):
            ss_list = [ss_list]

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
        self._info = pd.concat(
            [self._info] + [ss.info for ss in ss_list],
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

    def downsample_spectra(self, target_bins: int = 128):
        """Replaces spectra with downsampled version. Uniform binning is assumed.

        Args:
            target_bins: Defines into how many channels the current spectra channels
                should be combined.

        Raises:
            ValueError: Raised when binning is not a multiple of the target binning.

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
                transformation.T))

    def drop_sources_columns_with_all_zeros(self):
        """Removes columns from the sources DataFrame that contain only zeros.

            Modifications are made in-place.

        """
        self.sources = self.sources.loc[:, (self.sources != 0).any(axis=0)]

    def extend(self, spectra: Union[dict, list, np.array, pd.DataFrame],
               sources: pd.DataFrame, info: Union[list, pd.DataFrame]):
        """Extends the current SampleSet with the given data.

        Always verify that the data was appended in the way you expect based on what input type
        you are preferring to work with.

        Args:
            spectra: Defines the spectra to append to the current spectra DataFrame.
            sources: Defines the sources to append to the current sources DataFrame.
            info: Defines the info to append to the current info DataFrame.

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

    def get_all_channel_energies(self) -> np.ndarray:
        """Returns an array representing the energy (in keV) represented by the
        lower edge of each channel for all samples.

        Raises:
            ValueError: raised if any energy calibration terms are missing.

        """
        fractional_energy_bins = np.linspace(0, 1, self.n_channels)
        all_channel_energies = []
        for i in range(self.n_samples):
            channel_energies = self.get_channel_energies(i, fractional_energy_bins)
            all_channel_energies.append(channel_energies)
        return all_channel_energies

    def get_channel_energies(self, index, fractional_energy_bins=None) -> np.ndarray:
        """Returns an array representing the energy (in keV) represented by the
        lower edge of each channel for the sample at the specified index.

        Raises:
            ValueError: raised if any energy calibration terms are missing.

        """
        if fractional_energy_bins is None:
            fractional_energy_bins = np.linspace(0, 1, self.n_channels)

        # TODO: raise error if ecal info is missing for any row
        offset, gain, quadratic, cubic, low_energy = self.ecal[index]
        channel_energies = self._channels_to_energies(
            fractional_energy_bins,
            offset, gain, quadratic, cubic, low_energy
        )

        return channel_energies

    def get_labels(self, target_level="Isotope", max_only=True,
                   include_value=False, min_value=0.01,
                   level_aggregation="sum"):
        """Gets row labels for each spectrum based on source values.

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
                        level_aggregation="sum"):
        """Gets row labels for each spectrum based on prediction values.

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

    def get_samples(self):
        """Gets only the data (spectra + extra data) within the SampleSet
        as a NumPy array ready for use in model training.

        Returns:
            An ndarray containing the NaN-trimmed values for spectra and
            extra data.

        """
        spectra_values = np.nan_to_num(self._spectra)
        return spectra_values

    def get_source_contributions(self, target_level="Isotope"):
        """Gets the 2D array of values representing the percent contributions of each source.

        Returns:
            An ndarray containing the source contributions for each sample (i.e., ground truth).

        """
        return self.sources.groupby(axis=1, level=target_level).sum()

    def sources_columns_to_dict(self, target_level="Isotope") -> Union[dict, list]:
        """Converts the MultiIndex columns of the sources DataFrame to a dictionary.

        Note: depending on `target_level` and sources columns, duplicate values are possible.

        Args:
            target_level: the level of the MultiIndex at which the dictionary should start.
                If "Seed" is chosen, then a flat list will be returned.

        Returns:
            If `target_level` is "Category" or "Isotope," then a dict is returned.
            If `target_level` is "Seed," then a list is returned.

        Raises:
            ValueError: if `target_level` is invalid.
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

    def normalize(self, p: float = 1, clip_negatives: bool = True):
        """Normalizes spectra by L-p normalization.

        Default is L1 norm (p=1) can be interpreted as a forming a probability mass function (PMF).
        L2 norm (p=2) is unit energy (sum of squares == 1).

        Ref: "https://en.wikipedia.org/wiki/Parseval's_theorem"

        Args:
            p: Defines the exponent to which the spectra values are raised.
            clip_negatives: Determines whether or not negative values will
                be removed from the spectra and replaced with 0.

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
        """Converts sources to a valid probability mass function (PMF)."""
        self._sources = self._sources.divide(
            self._sources.sum(axis=1),
            axis=0
        )

    def replace_nan(self, replace_value: float = 0):
        """Replaces np.nan() values with replace_value.

        Args:
            replace_value: The value with which NaN will be replaced.

        """
        self._spectra.replace(np.nan, replace_value)
        self._sources.replace(np.nan, replace_value)
        self._info.replace(np.nan, replace_value)

    def sample(self, n_samples: int, random_seed: int = None) -> SampleSet:
        """Randomly samples the SampleSet.

        Args:
            n_samples: the number of random samples to be returned.
            random_seed: the seed value for random number generation.

        Returns:
            A sampleset of `n_samples` randomly selected measurements.

        """
        if random_seed:
            random.seed(random_seed)

        if n_samples > self.n_samples:
            n_samples = self.n_samples

        random_indices = random.sample(self.spectra.index.values.tolist(), n_samples)
        random_mask = np.isin(self.spectra.index, random_indices)
        return self[random_mask]

    def split_fg_and_bg(self, bg_seed_names: Iterable = DEFAULT_BG_SEED_NAMES) \
            -> Tuple[SampleSet, SampleSet]:
        """Splits the current SampleSet into two new SampleSets, one containing only foreground
        sources and the other containing only background sources.

        Foreground sources are assumed to be anything that is not designated as a background source.

        Args:
            bg_seeds_names: the names of the seeds which are considered background sources.
                This list be customized to also extract atypical background sources such as
                calibration sources.

        Returns:
            Two SampleSets, the first containing only foregrounds and the second only
            containing backgrounds.

        """
        seed_labels = self.get_labels(target_level="Seed")
        bg_mask = seed_labels.isin(bg_seed_names)

        fg_seeds_ss = self[~bg_mask]
        fg_seeds_ss.drop_sources_columns_with_all_zeros()
        bg_seeds_ss = self[bg_mask]
        bg_seeds_ss.drop_sources_columns_with_all_zeros()

        return fg_seeds_ss, bg_seeds_ss

    def to_hdf(self, path: str, verbose=False):
        """Writes the sampleset to disk as a HDF file at the given path.

        Args:
            path: the intended location and name of the resulting file.

        """
        if not path.lower().endswith(riid.SAMPLESET_FILE_EXTENSION):
            logging.warning(f"Path does not end in {riid.SAMPLESET_FILE_EXTENSION}")

        _write_hdf(self, path)
        if verbose:
            logging.info(f"Saved SampleSet to '{path}'")

    def to_pcf(self, path: str, verbose=False):
        """Writes the sampleset to disk as a PCF at the given path.

        Args:
            path: the intended location and name of the resulting file.

        """
        if not path.lower().endswith(riid.PCF_FILE_EXTENSION):
            logging.warning(f"Path does not end in {riid.PCF_FILE_EXTENSION}")

        _dict_to_pcf(_ss_to_pcf_dict(self), path)

        if verbose:
            logging.info(f"Saved SampleSet to '{path}'")

    def update_timestamp(self):
        """Sets the timestamp for all samples to now (UTC) in a standard format."""
        self.info.timestamp = datetime.utcnow().isoformat().replace(":", "_")

    def upsample_spectra(self, target_bins: int = 4096):
        """Replaces spectra with upsampled version. Uniform binning is assumed.

        Args:
            target_bins: Defines the number of bins into which the current spectra
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


def read_hdf(path: str) -> SampleSet:
    """Reads the HDF file in from the given filepath and creates
    a SampleSet object with the data.

    Args:
        path: Defines the path to the file to be read in.

    Returns:
        A SampleSet object.

    """
    ss = None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No file found at location '{path}'.")

    ss = _read_hdf(path)

    if not ss:
        raise FileNotFoundError(f"No data found at location '{path}'.")

    return ss


def read_pcf(path: str, verbose: bool = False) -> SampleSet:
    """Converts pcf file to sampleset object.

    Args:
        path: Defines the path to the PCF file to be read in.
        verbose: Determines whether or not to show verbose function output in terminal.

    Returns:
        A Sampleset object containing information of pcf file in sampleset format

    Raises:
        None.
    """
    return _pcf_dict_to_ss(_pcf_to_dict(path, verbose), verbose)


def _get_row_labels(df: pd.DataFrame, target_level: str = "Isotope", max_only: bool = True,
                    include_value: bool = False, min_value: float = 0.01,
                    level_aggregation: str = "sum") -> pd.Series:
    """Interprets the cell values in conjunction with the columns to determine an
        appropriate label for each row.

        Args:
            df: The DataFrame with MultiIndex columns and named levels.
            target_level: the level of the columns to use for row labels.
            max_only: for each row, whether or not to only use the column
                containing the max value.  When False, all column names
                containing a cell value >= `min_value` will be obtained.
            include_value: whether or not to include the cell value
                alongside the column name(s).
            min_value: when `max_only=False`, this filters out columns
                containing values less than this value. Note: this applies
                to values AFTER aggregation occurs.
            level_aggregation: the method to use for combining
                cell values which have the same column name at the
                `target_level`.  Acceptable values are "sum" or "mean",
                otherwise this is ignored.

        Returns:
            A Pandas Series where each entry is the row label.
    """
    if max_only:
        if level_aggregation == "sum":
            values = df.groupby(axis=1, level=target_level).sum()
            labels = values.idxmax(axis=1)
        elif level_aggregation == "mean":
            values = df.groupby(axis=1, level=target_level).mean()
            labels = values.idxmax(axis=1)
        else:
            values = df
            labels = pd.MultiIndex.from_tuples(
                values.idxmax(axis=1),
                names=SampleSet.SOURCES_MULTI_INDEX_NAMES
            ).get_level_values(target_level)
        if include_value:
            values = values.max(axis=1)
            labels = [f"{x} ({y:.2f})" for x, y in zip(labels, values)]
    else:  # Much slower
        if level_aggregation == "sum":
            values = df.groupby(axis=1, level=target_level).sum()
        elif level_aggregation == "mean":
            values = df.groupby(axis=1, level=target_level).mean()
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
    """Validates the SampleSet keys based on whether or not
    it contains the minimum required set of keys. Used when creating
    a SampleSet from a file.

    Args:
        keys: Defines the collection of keys from the dataset being
            converted to a SampleSet.

    Raises:
        InvalidSampleSetFileError: Raised when the set of keys provided
            does not include a required key.

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


def _read_hdf(file_name: str) -> SampleSet:
    """Reads sampleset class from hdf binary format.

    Args:
        file_name: Defines the string representing the relative
            path to the HDF file.

    Returns:
        A SampleSet containing the HDF data.

    """
    store = pd.HDFStore(file_name, mode="r")
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
    if "measured_or_synthetic" in other_info:
        ss.measured_or_synthetic = other_info["measured_or_synthetic"].iloc[0]
    if "detector_info" in other_info:
        ss.detector_info = other_info["detector_info"].iloc[0]
    if "synthesis_info" in other_info:
        ss.synthesis_info = other_info["synthesis_info"].iloc[0]
    if "classified_by" in other_info:
        ss.classified_by = other_info["classified_by"].iloc[0]

    return ss


def _write_hdf(ss: SampleSet, output_path: str):
    """Writes sampleset class to hdf binary format.

    Args:
        ss: Defines the SampleSet object to be written out.
        output_path: Defines the relative path to the file to
            be written.

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        store = pd.HDFStore(output_path, "w")
        store.put("spectra", ss.spectra)
        store.put("sources", ss.sources)
        store.put("info", ss.info)
        store.put("prediction_probas", ss.prediction_probas)

        other_info = {
            "measured_or_synthetic": ss.measured_or_synthetic,
            "detector_info": ss.detector_info,
            "synthesis_info": ss.synthesis_info,
            "classified_by": ss.classified_by
        }
        store.put("other_info", pd.DataFrame(data=[other_info]))
        store.close()


def _ss_to_pcf_dict(ss: SampleSet):
    """Converts a Sampleset to a dictionary of values.

    Args:
        ss: Defines a Sampleset object to be converted.

    Returns:
        A dictionary containing the values from the Sampleset.

    Raises:
        None.
    """
    n_channels = ss.n_channels
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

    for i in range(ss.n_samples):
        title = isotopes[i] if not isotopes.empty else NO_ISOTOPE
        description = ss.info.description.fillna("").iloc[i]
        source = seeds[i] if not seeds.empty else NO_SEED
        compressed_text_buffer = _pack_compressed_text_buffer(title, description, source)

        header = {
            "Compressed_Text_Buffer": compressed_text_buffer,
            "Energy_Calibration_Low_Energy": ss.info.ecal_low_e.fillna(0).iloc[i],
            "Energy_Calibration_Offset": ss.info.ecal_order_0.fillna(0).iloc[i],
            "Energy_Calibration_Gain": ss.info.ecal_order_1.fillna(0).iloc[i],
            "Energy_Calibration_Quadratic": ss.info.ecal_order_2.fillna(0).iloc[i],
            "Energy_Calibration_Cubic": ss.info.ecal_order_3.fillna(0).iloc[i],
            "Live_Time": ss.info.live_time.fillna(0).iloc[i],
            "Total_time_per_real_time": ss.info.real_time.fillna(0).iloc[i],
            "Number_of_Channels": int(n_channels),
            "Date-time_VAX": ss.info.timestamp.fillna("").iloc[i],
            "Occupancy_Flag": ss.info.occupancy_flag.fillna(0).iloc[i],
            "Tag": ss.info.tag.fillna("").iloc[i],
            "Total_Neutron_Counts": ss.info.neutron_counts.fillna(0).iloc[i],
        }

        spectrum = ss.spectra.values[i, :]
        spectra.append({"header": header, "spectrum": spectrum})
    return {"header": pcf_header, "spectra": spectra}


def _pcf_dict_to_ss(pcf_dict: dict, verbose=True):
    """Converts pcf dictionary into a SampleSet.

    Args:
        pcf_dict: Defines the dictionary of pcf values.
        verbose: Whether to display output from attempting the conversion.

    Returns:
        A Sampleset object containing the pcf dict values.

    Raises:
        None.
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

        # PCF file contains energy calibration terms which are defined as:
        # E_i = a0 + a1*x + a2*x^2 + a3*x^3 + a4 / (1 + 60*x)
        # where:
        #   a0 = order_0
        #   a1 = order_1
        #   a2 = order_2
        #   a3 = order_3
        #   a4 = low_E
        #   x = channel number
        #   E_i = Energy value of i"th channel

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
            # The following commented out fields are PyRIID-only, which can't be stored in PCF.
            # They are shown here to explicitly communicate what would be lost in translation.
            #   - snr_target
            #   - snr_estimate
            #   - sigma
            #   - bg_counts
            #   - fg_counts
            #   - bg_counts_expected
            #   - fg_counts
            "total_counts": sum(spectrum["spectrum"]),
            "total_neutron_counts": spectrum["header"]["Total_Neutron_Counts"],
            "distance_cm": distance,
            "area_density": ad,
            "ecal_order_0": order_0,
            "ecal_order_1": order_1,
            "ecal_order_2": order_2,
            "ecal_order_3": order_3,
            "ecal_low_e": low_E,
            "atomic_number": an,
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
        sources_df[key].iloc[i] = 1.0

    ss = SampleSet()
    ss.spectra = pd.DataFrame(data=spectra)
    ss.sources = sources_df
    ss.info = pd.DataFrame(data=infos)
    ss.measured_or_synthetic = "synthetic",
    ss.detector_info["pcf_metadata"] = pcf_dict["header"]

    return ss


class RebinningCalculationError(Exception):
    """An exception that indicates an issue when rebinning
    (up-binning or down-binning) a sample or collection of samples.
    """
    pass


class InvalidSampleSetFileError(Exception):
    """An exception that indicates missing or invalid keys
    in a file being parsed into a SampleSet.
    """
    pass
