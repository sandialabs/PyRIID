# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This module contains the SampleSet class."""
import copy
import logging
import random

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class SampleSet():
    """Container for collection of samples.
    """
    # pylint: disable=R0902
    # pylint: disable=R0904

    def __init__(self, **kwargs):
        # Give all class attributes some default values
        self._spectra = pd.DataFrame()  # expected to be of size n_samples X n_channels
        # expected to be of size n_samples X (n_source_types + 1) Labels Column
        # will exist
        self._sources = pd.DataFrame()
        # expected to be of size n_samples X (n feature columns defined)
        self._features = pd.DataFrame()
        self._detector = None
        df_columns = [
            "live_time",
            "snr_target",
            "snr_estimate",
            "bg_counts",
            "fg_counts",
            "bg_counts_expected",
            "total_counts",
            "sigma",
            "distance",
            "atomic_number",
            "area_density",
            "ecal_order_0",
            "ecal_order_1",
            "ecal_order_2",
            "ecal_order_3",
            "ecal_low_e",
            "date-time",
            "real_time",
            "occupancy_flag",
            "tag",
            "total_neutron_counts",
            "descr"
        ]
        self._collection_information = pd.DataFrame(columns=df_columns)
        self._sensor_information = {}
        self._config = None
        self._prediction_probas = pd.DataFrame(np.array([]))
        self._predictions = np.array([])
        self._measured_or_synthetic = None
        self._subtract_background = None
        self._purpose = None
        self._comments = ""
        self._energy_bin_centers = np.array([])
        self._energy_bin_edges = np.array([])
        self._pcf_metadata = None
        self.detector_hash = ""
        self.neutron_detector_hash = ""

        # Populate values passed in.
        if kwargs:
            for key, value in kwargs.items():
                self[key] = value

        # Check if updated values were missing fields.
        missing_columns = [column for column in df_columns if
                           column not in self._collection_information.columns]
        if missing_columns:
            self._collection_information = pd.concat(
                    (
                        self._collection_information,
                        pd.DataFrame(columns=missing_columns)
                    ),
                    sort=False
            )

    def __str__(self):
        return "Sampleset for {} containing {} samples with {} channels.".format(
            self.detector,
            self.n_samples,
            self.n_channels
        )

    def __getitem__(self, key):
        item = None
        try:
            if key == "sources":
                item = self.sources.replace(np.nan, 0)
            else:
                item = getattr(self, key)
        except KeyError as error:
            msg = "{0}".format(error)
            print(msg)
        return item

    def __setitem__(self, key, value):
        try:
            setattr(self, key, value)
        except KeyError as error:
            msg = "{0}".format(error)
            print(msg)

    def concat(self, ss_list):
        """ Provides method of combining many samplesets quickly
        """
        if not isinstance(ss_list, list):
            ss_list = [ss_list]

        all_spectra = [self.spectra]
        for ss in ss_list:
            spectra = ss.spectra
            all_spectra.append(spectra)
        self._spectra = pd.concat(all_spectra, ignore_index=True, sort=False)

        orig_sources = self.label_matrix
        orig_sources.loc[:, "label"] = self.labels
        all_sources = [orig_sources]
        for ss in ss_list:
            sources = ss.label_matrix
            sources.loc[:, "label"] = ss.labels
            all_sources.append(sources)

        self._sources = pd.concat(all_sources, ignore_index=True, sort=False)
        self._sources.replace(np.nan, 0.0, inplace=True)

        all_collection_information = [self._collection_information]
        for ss in ss_list:
            all_collection_information.append(ss._collection_information)
        self._collection_information = pd.concat(all_collection_information, ignore_index=True, sort=False)

        self._prediction_probas = pd.DataFrame(np.array([]))
        self._predictions = np.array([])

    def to_pmf(self, clip_negatives=True):
        """ Converts spectra to probability mass function (PMF)
        """
        if clip_negatives:
            self._spectra = self._spectra.clip(0)

        if (self._spectra.values < 0).sum():
            logging.warning(
                "You are performing a PMF operation on spectra which contains negative values.  " +
                "Consider applying `clip_negatives`."
            )

        self._spectra = pd.DataFrame(
            data=self._spectra.values / self._spectra.values.sum(axis=1)[:, np.newaxis]
        )

    def pmf_to_counts(self, verbose=0):
        """Converts spectra from PMF to counts.
        """
        row_sums = self._spectra.values.sum(axis=1)
        rows_eq_one = np.where(abs(row_sums - 1.0) <= 1e-5)[0]
        rows_neq_one = np.where(abs(row_sums - 1.0) > 1e-5)[0]
        for idx in rows_eq_one:
            self._spectra.iloc[idx] = \
                self._spectra.iloc[idx] * self._collection_information.loc[idx, "total_counts"]

        logging.info("PMF to Counts Conversion Summary:")
        logging.info(f"  Spectra converted:         {len(rows_eq_one)}")
        if rows_neq_one.size != 0:
            logging.info(f"  Spectra not converted:     {rows_neq_one.size}")
            if verbose:
                logging.info(f"      {rows_neq_one}")
                logging.info(f"      {row_sums[rows_neq_one]}")

    def to_count_rate(self):
        """ Converts spectra to count rage (live_time normalized)
        """
        self._spectra = pd.DataFrame(
            data=self._spectra.values / self.live_time[:, np.newaxis])

    def clip_negatives(self, min_value=0):
        """ Sets negative values to min_value
        """
        self._spectra = pd.DataFrame(data=self._spectra.clip(min_value))

    def replace_na(self, replace_value=0):
        """ Replaces np.nan() values with replace_value.
        """
        self._spectra.replace(np.nan, replace_value)
        self._sources.replace(np.nan, replace_value)
        self._collection_information.replace(np.nan, replace_value)

    def downsample_spectra(self, target_bins=128, binning_type="uniform"):
        """ Replaces spectra with downsampled version.  Uniform binning is assumed.
        """
        if binning_type == "uniform":
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
            self._energy_bin_centers = self._get_energy_centers()
        elif binning_type == "sqrt":
            n_source = self.n_channels

            max_n_per = (n_source / target_bins - 1) * 2 + 1
            n_per = np.round(np.linspace(1, max_n_per, target_bins))
            n_off = int(n_source - n_per.sum())

            if n_off > 0:
                n_per[-n_off:] += 1
            elif n_off < 0:
                n_per[n_off:] -= 1
            n_per = n_per.clip(1)

            if n_per.sum() != n_source:
                raise RebinningCalculationError("Rebinning calculation implemented incorrectly.")

            n_per = n_per.astype(int)
            transformation = np.zeros([n_source, target_bins])
            i_start = 0
            for col, number_of_channels in enumerate(n_per):
                i_finish = i_start + number_of_channels
                transformation[i_start:i_finish, col] = 1
                i_start = i_finish
            self._spectra = pd.DataFrame(
                data=np.matmul(
                    self._spectra.values,
                    transformation))
            self._energy_bin_centers = self._get_energy_centers()
        else:
            msg = "Binning must be 'uniform' or 'sqrt', '{}' was entered.".format(binning_type)
            raise ValueError(msg)

    def upsample_spectra(self, target_bins=4096):
        """ Replaces spectra with upsampled version.  Uniform binning is assumed.
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

    def relabel_from_dict(self, mapping_dict, mode="both"):
        """ Relabels labels and predictions according to "mapping_dict".
        "mapping_dict" should have "source":"target_label" pairs.

        mode="all: performs relableing of "label" and "label_matrix_label"
        mode="single": performs relabeling of "label" column of "sources" DataFrame
        mode="multi": performs relabeling of label_matrix columns of "sources" DataFrame
        """

        if mode not in ["both", "single", "multi"]:
            raise ValueError("Mode " + mode + " must be in set {'all', 'single', 'multi'}.")

        if mode in ["both", "single"]:
            for label in set(self.labels):
                if label not in mapping_dict.keys():
                    msg = str(label)
                    msg += " from single label was not specified in mapping dictionary."
                    raise MissingKeyForRelabeling(msg)
            orig_single_labels = self.labels
            self.labels = [mapping_dict[key] for key in orig_single_labels]

        if mode in ["both", "multi"]:
            for label in self.label_matrix_labels:
                if label not in mapping_dict.keys():
                    msg = str(label)
                    msg += " from label_matrix_labels was not specified in mapping dictionary."
                    raise MissingKeyForRelabeling(msg)
            orig_multi_labels = self.label_matrix_labels
            self.label_matrix_labels = [mapping_dict[key] for key in orig_multi_labels]

    def relabel_to_max_source(self):
        """ Relabels labels to the source name of most contributing source.
        """

        self.labels = self.label_matrix_labels[self.label_matrix.values.argmax(axis=1)]

    def sample(self, n_samples, random_seed=None, reindex=True):
        """ Returns n random observations.
        """
        if random_seed is not None:
            random.seed(random_seed)

        if n_samples > self.n_samples:
            n_samples = self.n_samples
        indices = random.sample(self.spectra.index.values.tolist(), n_samples)
        return self.get_indices(np.isin(self.spectra.index, indices), reindex=reindex)

    def shuffle(self, random_seed=None):
        """ Shuffles observations.
        """
        if random_seed is not None:
            random.seed(random_seed)
        indices = random.sample(self.spectra.index.values.tolist(), self.n_samples)
        self.sources = self.sources.iloc[indices, :]
        self.spectra = self.spectra.iloc[indices, :]
        self.collection_information = self.collection_information.iloc[indices, :]

    def split(self, split_ratio=0.2, random_seed=None):
        """ Returns n random observations.
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        indices = self.spectra.index.values
        np.random.shuffle(indices)
        val_mask = np.full(len(indices), False)
        train_mask = np.full(len(indices), True)

        n_val = int(len(indices) * split_ratio)
        val_indices = np.random.choice(np.arange(len(indices)), n_val)
        val_mask[val_indices] = True
        train_mask[val_indices] = False

        train_ss = self.get_indices(train_mask)
        val_ss = self.get_indices(val_mask)

        return train_ss, val_ss

    def get_indices(self, indices, reindex=True):
        """ Returns SampleSet of only the observations defined by "indices".
        """
        if isinstance(indices, pd.core.series.Series):
            indices = indices.values
        samples = copy.copy(self)
        samples.sources = samples.sources.loc[indices, :]
        samples.spectra = samples.spectra.loc[indices, :]
        samples.collection_information = samples.collection_information.loc[indices, :]
        if samples.predictions.shape[0]:
            samples.predictions = samples.predictions[indices]
        if samples.prediction_probas.shape[0]:
            try:
                samples.prediction_probas = samples.prediction_probas.loc[indices, :]
            except AttributeError:
                samples.prediction_probas = samples.prediction_probas[indices, :]

        if samples.features.shape[0]:
            try:
                samples.features = samples.features.loc[indices, :]
            except AttributeError:
                samples.features = samples.features[indices, :]

        if reindex:
            samples.sources.index = np.arange(samples.n_samples)
            samples.spectra.index = np.arange(samples.n_samples)
            samples.collection_information.index = np.arange(samples.n_samples)
        return samples

    def get_features(self, include_spectra=True):
        """get_features getter"""
        if include_spectra:
            if self._features.shape[0] > 0:
                if self._features.ndim == 1:
                    floc = np.expand_dims(self._features, 1)
                else:
                    floc = self._features
                return np.nan_to_num(
                    np.concatenate(
                        (self._spectra, floc), axis=1))
            else:
                return np.nan_to_num(self._spectra.values)
        else:
            return np.nan_to_num(self._features.values)

    def to_energy(self, energy_bin_centers, inbin_values=None):
        """ Transforms specrum into energy units centered at
        "energy_bin_centers"
        """
        orig_counts = self.spectra.sum(axis=1)
        energy_spectra = np.zeros([self.n_samples, len(energy_bin_centers)])
        original_energy_centers = self._get_energy_centers(inbin_values)

        loop_values = zip(original_energy_centers, self.spectra.values)
        for j, (old_centers, old_values) in enumerate(loop_values):
            interp_func = interp1d(
                old_centers,
                old_values,
                kind="linear",
                fill_value=0,
                bounds_error=False
            )
            new_spectrum = interp_func(energy_bin_centers)
            energy_spectra[j, :] = new_spectrum
        # Normalize to maintain total counts
        energy_spectra = energy_spectra.clip(0)
        total_counts = energy_spectra.sum(axis=1)
        total_counts[total_counts <= 0] = 1
        energy_spectra = (energy_spectra / total_counts[:, None]) *\
            orig_counts[:, None]
        self.spectra = pd.DataFrame(energy_spectra).replace(np.nan, 0)

    def _get_energy_centers(self, bins=None):
        """  Returns an array of the energy centers for the calibration of each
        spectra in the SampleSet.
        """
        if bins is None:
            bins = np.linspace(0, 1, self.n_channels)

        energy_centers = self.collection_information.ecal_order_0.values[:, None] + \
            bins * self.collection_information.ecal_order_1.values[:, None] + \
            bins**2 * self.collection_information.ecal_order_2.values[:, None] + \
            bins**3 * self.collection_information.ecal_order_3.values[:, None] + \
            self.collection_information.ecal_low_e.values[:, None] / (1 + 60 * bins)
        return energy_centers

    def _combine_redundant_sources(self):
        """Combines redundant columns of sources DataFrame.
        """
        columns = self._sources.columns.values
        self._sources = self._sources.groupby(columns, axis=1, sort=False).sum()

    def add_spectra(self, spectra, sources, labels, info):
        """Appends the given arguments to their respective dataframes.
        """
        self._spectra = self._spectra.append(pd.DataFrame(spectra), ignore_index=True, sort=True)
        self._sources = self._sources.append(pd.DataFrame(sources), ignore_index=True, sort=True)
        self._sources["label"] = labels
        self._collection_information = self._collection_information.append(
            pd.DataFrame(info),
            ignore_index=True,
            sort=True
        )

    def add_spectrum(self, spectrum, source_values, label):
        """Appends the list-like spectrum to the spectra and sources DataFrame.
        """
        if not isinstance(source_values, dict):
            raise TypeError("'source_values' argument must be of type 'dict'")
        if not all([isinstance(x, str) for x in source_values.keys()]):
            raise TypeError("Keys in 'source_values' argument must all be of type 'str'")
        if not all([isinstance(x, float) for x in source_values.values()]):
            raise TypeError("Values in 'source_values' argument must all be of type 'float'")
        if sum(source_values.values()) != 1.0:
            raise ValueError("Values in 'source_values' argument must sum to '1.0'")
        # TODO: check spectrum type and length against existing spectra rows

        self._spectra = self._spectra.append(pd.Series(spectrum), ignore_index=True)
        self._sources = self._sources.append(source_values, ignore_index=True)
        self._sources.at[self._sources.index[-1], "label"] = label

    def add_info(self, live_time=0.0, snr_target=0.0, snr_estimate=0.0, bg_counts=0, fg_counts=0,
                 bg_counts_expected=0, total_counts=0, sigma=0, ecal_order_0=0,
                 ecal_order_1=0.0, ecal_order_2=0.0, ecal_order_3=0.0, ecal_low_e=0.0,
                 real_time=0.0, occupancy_flag=False, total_neutron_counts=0, tag="",
                 date_time=None, desc="", **kwargs):
        """Appends the provide values to the collection information DataFrame.
        """
        row = {
            "live_time": live_time,
            "snr_target": snr_target,
            "snr_estimate": snr_estimate,
            "bg_counts": bg_counts,
            "fg_counts": fg_counts,
            "bg_counts_expected": bg_counts_expected,
            "total_counts": total_counts,
            "sigma": sigma,
            "ecal_order_0": ecal_order_0,
            "ecal_order_1": ecal_order_1,
            "ecal_order_2": ecal_order_2,
            "ecal_order_3": ecal_order_3,
            "ecal_low_e": ecal_low_e,
            "real_time": real_time,
            "occupancy_flag": occupancy_flag,
            "total_neutron_counts": total_neutron_counts,
            "tag": tag,
            "date-time": date_time,
            "descr": desc
        }
        if kwargs:
            for key, value in kwargs.items():
                row[key] = value
        self._collection_information = self._collection_information.append(row, ignore_index=True)

    @property
    def spectra(self):
        """spectra getter setter"""
        return self._spectra

    @spectra.setter
    def spectra(self, value):
        self._spectra = value

    @property
    def sources(self):
        """sources getter setter"""
        return self._sources

    @sources.setter
    def sources(self, value):
        """sources setter"""
        self._sources = value.replace(np.nan, 0)
        self._combine_redundant_sources()

    @property
    def label_matrix_labels(self):
        """Getter for label matrix labels"""
        return self.label_matrix.columns.values

    @label_matrix_labels.setter
    def label_matrix_labels(self, values):
        """Setter for label matrix labels"""
        columns = copy.deepcopy(list(values))
        columns.append("label")
        self._sources.columns = columns
        self._combine_redundant_sources()

    @property
    def features(self):
        """features getter setter"""
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def detector(self):
        """detector getter setter"""
        return self._detector

    @detector.setter
    def detector(self, value):
        self._detector = value

    @property
    def purpose(self):
        """purpose getter setter"""
        return self._purpose

    @purpose.setter
    def purpose(self, value):
        self._purpose = value

    @property
    def distance(self):
        """distance getter setter"""
        return self._collection_information.distance

    @distance.setter
    def distance(self, value):
        self._collection_information.loc[:, "distance"] = value

    @property
    def collection_information(self):
        """collection_information getter setter"""
        return self._collection_information

    @collection_information.setter
    def collection_information(self, value):
        self._collection_information = value

    @property
    def sensor_information(self):
        """sensor_information getter setter"""
        return self._sensor_information

    @sensor_information.setter
    def sensor_information(self, value):
        self._sensor_information = value

    @property
    def config(self):
        """config getter setter"""
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @property
    def prediction_probas(self):
        """prediction_probas getter setter"""
        return self._prediction_probas

    @prediction_probas.setter
    def prediction_probas(self, value):
        if isinstance(value, pd.DataFrame):
            self._prediction_probas = value
        else:
            self._prediction_probas = pd.DataFrame(value, columns=self.label_matrix_labels)

    @property
    def snr_estimate(self):
        """snr_estimate getter setter"""
        return self._collection_information.snr_estimate

    @snr_estimate.setter
    def snr_estimate(self, value):
        self._collection_information.loc[:, "snr_estimate"] = value

    @property
    def predictions(self):
        """predictions getter setter"""
        return np.array(self._predictions)

    @predictions.setter
    def predictions(self, value):
        self._predictions = value

    @property
    def measured_or_synthetic(self):
        """measured_or_synthetic getter setter"""
        return self._measured_or_synthetic

    @measured_or_synthetic.setter
    def measured_or_synthetic(self, value):
        self._measured_or_synthetic = value

    @property
    def subtract_background(self):
        """subtract_background getter setter"""
        return self._subtract_background

    @subtract_background.setter
    def subtract_background(self, value):
        self._subtract_background = value

    @property
    def comments(self):
        """comments getter setter"""
        return self._comments

    @comments.setter
    def comments(self, value):
        self._comments = value

    # Repeat the below with the rest of the columns of the collection info table
    @property
    def live_time(self):
        """live_time getter setter"""
        return self._collection_information.live_time

    @live_time.setter
    def live_time(self, value):
        self._collection_information.loc[:, "live_time"] = value

    @property
    def sigma(self):
        """sigma getter setter"""
        return self._collection_information.sigma

    @sigma.setter
    def sigma(self, value):
        self._collection_information.loc[:, "sigma"] = value

    @property
    def total_counts(self):
        """total_counts getter setter"""
        return self._collection_information.total_counts

    @total_counts.setter
    def total_counts(self, value):
        self._collection_information.loc[:, "total_counts"] = value

    @property
    def labels(self):
        """labels getter setter"""
        return self._sources.loc[:, "label"].values

    @labels.setter
    def labels(self, value):
        self._sources.loc[:, "label"] = value

    @property
    def label_matrix(self):
        """label_matrix getter"""
        return self._sources.loc[:, self._sources.columns != "label"].replace(np.nan, 0)

    @property
    def n_samples(self):
        """n_samples getter"""
        return self._spectra.shape[0]

    @property
    def n_channels(self):
        """n_channels getter"""
        return self._spectra.shape[1]

    @property
    def n_features(self, include_spectra=True):
        """n_features getter"""
        if include_spectra:
            return self.n_channels + self._features.shape[1]
        else:
            return self._features.shape[1]

    @property
    def source_types(self):
        """source_types getter"""
        return self.label_matrix.columns.values

    @property
    def energy_bin_centers(self):
        """energy_bin_centers getter"""
        if not len(self._energy_bin_centers):
            self._energy_bin_centers = self._get_energy_centers()
        return self._energy_bin_centers

    @energy_bin_centers.setter
    def energy_bin_centers(self, value):
        """energy_bin_centers setter"""
        self._energy_bin_centers = value

    @property
    def ecal_order_0(self):
        """ecal_order_0 getter"""
        return self.collection_information.ecal_order_0

    @ecal_order_0.setter
    def ecal_order_0(self, value):
        """ecal_order_0 setter"""
        self.collection_information.loc[:, "ecal_order_0"] = value

    @property
    def ecal_order_1(self):
        """ecal_order_1 getter"""
        return self.collection_information.ecal_order_1

    @ecal_order_1.setter
    def ecal_order_1(self, value):
        """ecal_order_1 setter"""
        self.collection_information.loc[:, "ecal_order_1"] = value

    @property
    def ecal_order_2(self):
        """ecal_order_2 getter"""
        return self.collection_information.ecal_order_2

    @ecal_order_2.setter
    def ecal_order_2(self, value):
        """ecal_order_2 setter"""
        self.collection_information.loc[:, "ecal_order_2"] = value

    @property
    def ecal_order_3(self):
        """ecal_order_3 getter"""
        return self.collection_information.ecal_order_3

    @ecal_order_3.setter
    def ecal_order_3(self, value):
        """ecal_order_3 setter"""
        self.collection_information.loc[:, "ecal_order_3"] = value

    @property
    def ecal_low_e(self):
        """ecal_low_e getter"""
        return self.collection_information.ecal_low_e

    @ecal_low_e.setter
    def ecal_low_e(self, value):
        """ecal_low_e setter"""
        self.collection_information.loc[:, "ecal_low_e"] = value

    @property
    def ecal_factors(self):
        """ecal_factors getter
        returns calibration factors in following order:
            order_0, order_1, order_2, order_3, low_e
        """
        calibration_factors = np.array([
            self.collection_information.ecal_order_0.values,
            self.collection_information.ecal_order_1.values,
            self.collection_information.ecal_order_2.values,
            self.collection_information.ecal_order_3.values,
            self.collection_information.ecal_low_e.values
        ])
        return calibration_factors

    @ecal_factors.setter
    def ecal_factors(self, calibration_factors):
        """ecal_low_e setter"""
        self.collection_information.loc[:, "ecal_order_0"] = calibration_factors[0]
        self.collection_information.loc[:, "ecal_order_1"] = calibration_factors[1]
        self.collection_information.loc[:, "ecal_order_2"] = calibration_factors[2]
        self.collection_information.loc[:, "ecal_order_3"] = calibration_factors[3]
        self.collection_information.loc[:, "ecal_low_e"] = calibration_factors[4]

    @property
    def neutron_counts(self):
        """ neutron counts for collection
        """
        return self.collection_information.total_neutron_counts

    @neutron_counts.setter
    def neutron_counts(self, neutron_counts):
        """ setter for neutron counts
        """
        self.collection_information.total_neutron_counts = neutron_counts


class RebinningCalculationError(Exception):
    pass


class MissingKeyForRelabeling(Exception):
    pass
