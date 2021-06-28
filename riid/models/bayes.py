# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This module contains the Poisson Bayes classifier."""
import os

import numpy as np
import pandas as pd
from scipy.stats import poisson
from tqdm import tqdm

from riid.sampleset import SampleSet


class PoissonBayes:
    """This class implements a Bayes classifier for poisson distributed data."""

    def __init__(self, seeds: SampleSet = None):
        """Initializes the classifier with seeds.

        This constructor takes "seed" spectra as input to be used as the shapes against which
        classifications will be performed. It will apply a PMF to the SampleSet.

        Args:
            seeds: an optional SampleSet of seed spectra for each potential source of interest.
                A background seed may be included if classification as background is desired.

        Returns:
            None

        Raises:
            NegativeSpectrumError: raised if any seed spectrum has negative counts in any bin.
            ZeroTotalCountsError: raised if any seed spectrum contains zero total counts.
        """
        self._temp_file_path = "temp.mdl"

        if not seeds:
            return

        if seeds.n_samples <= 0:
            raise ValueError("At least one seed must be provided.")
        if (seeds.spectra.values < 0).any():
            msg = "Argument 'seeds' can't contain any spectra with negative values."
            raise NegativeSpectrumError(msg)
        if (seeds.spectra.values.sum(axis=1) <= 0).any():
            msg = "Argument 'seeds' can't contain any spectra with zero total counts."
            raise ZeroTotalCountsError(msg)

        seeds.to_pmf()
        self._seeds = seeds.spectra
        self._classes = seeds.labels
        self._n_bins = self._seeds.shape[1]

    def _calculate_valid_bins(self, bg_spectrum: np.ndarray) -> np.ndarray:
        """Calculates which bins are valid for use in classification based on the values present in
        both the seeds and the given background spectrum.

        Args:
            bg_spectrum: a numpy.ndarray representing a single background spectrum.
                Shape: (n_bins,)

        Returns:
            A numpy.ndarray of booleans representing which channels are valid (True) and which
            are not (False).

        Raises:
            None
        """
        return ((self._seeds.values.T + bg_spectrum[:, None]) > 0).prod(axis=1).astype(bool)

    def _predict_single(self, seed_spectra: np.ndarray, gross_spectrum: np.ndarray,
                        bg_spectrum: np.ndarray, estimated_bg_counts: float,
                        normalize_scores: bool = False) -> np.ndarray:
        """Performs a prediction of the probabilities of each class for single spectrum.

        Performs prediction on the given spectrum and returns the probabilies of observing the
        spectrum from each of the underlying classes.
        Each probability represents the chances that you would have measured the the given spectrum
        with the underlying distribution of a particular seed.
        Note that these probabilities are almost always extremely low (which actually makes sense
        since all of this is technically very unlikely).

        Args:
            seed_spectra: a numpy.ndarray containing the seed spectra on which to match.
                1D shape: (n-seeds, n_bins).
            gross_spectrum: a numpy.ndarray containing the spectrum needing to be identified.
                1D shape: (n_bins,).
            bg_spectrum: a numpy.ndarray containing the background spectrum associated with the
                gross spectrum. 1D shape: (n_bins,).
            estimated_bg_counts: an estimate of the number of counts from background sources.
            normalize_scores: when True scores will be normalized based upon the highest
                achievable score (perfect template).

        Returns:
            Returns a numpy.ndarray contining the probabilities for each class.
            1D shape: (n_classes,)

        Raises:
            None
        """
        total_counts = gross_spectrum.sum()
        estimated_fgs = np.abs((total_counts - estimated_bg_counts) * seed_spectra)
        estimated_bg = estimated_bg_counts * bg_spectrum
        expected_signal = estimated_bg[None, :] + estimated_fgs
        if normalize_scores:
            expected_signal = np.vstack(
                (
                    expected_signal,
                    gross_spectrum,
                )
            )

        probas = poisson.logpmf(gross_spectrum, expected_signal).mean(axis=1)

        if normalize_scores:
            probas = probas[:-1] - probas[-1]

        return probas

    def predict(self, gross_ss: SampleSet, bg_ss: SampleSet,
                normalize_scores: bool = False, verbose=0) -> list:
        """Attempts to classify the provided gross spectra using the seeds with which the
        PoissonBayes object was originally initialized.

        Args:
            gross_ss: a SampleSet of `n` gross spectra where `n` >= 1.
            bg_ss: a SampleSet of one or `n` background spectra.
                If a SampleSet with one background spectrum is provided, that background will be
                applied to all gross spectra. Otherwise, the number of background spectra must
                equal the number of gross spectra. In the latter case, it is assumed that the
                index positions of corresponding background and gross spectra are the same.
            normalize_scores: when True scores will be normalized based upon the highest
                achievable score (perfect template).

        Returns:
            None

        Raises:
            ValueError: Raised when:
                - no gross spectra are provided.
                - no background(s) are provided.
                - an invalid number of backgrounds is provided.
                - the binning of the provided SampleSets do not match the binning of
                  the previously provided seed spectra.
        """
        if gross_ss.n_samples <= 0:
            raise ValueError("No gross spectr[a|um] provided!")
        if bg_ss.n_samples <= 0:
            raise ValueError("No background spectr[a|um] provided!")
        if bg_ss.n_samples != 1 and bg_ss.n_samples != gross_ss.n_samples:
            msg = "The number of background spectra is not 1 (it's {}).  ".format(bg_ss.n_samples)
            msg += "Therefore you must provide equal numbers of background and gross spectra!"
            raise ValueError(msg)
        if gross_ss.n_channels != self._n_bins:
            msg = "The binning of all gross spectra must be the same as the provided seeds!  "
            msg += "Seed spectra contain {} bins.".format(self._n_bins)
            raise ValueError(msg)
        if bg_ss.n_channels != self._n_bins:
            msg = "The binning of all background spectra must be the same as the provided seeds!  "
            msg += "Seed spectra all contain {} bins.".format(self._n_bins)
            raise ValueError(msg)

        bg_ss.to_pmf()
        probas = np.zeros([gross_ss.spectra.values.shape[0], self._classes.shape[0]])
        for i in tqdm(range(gross_ss.n_samples), desc="Samples", disable=not verbose, leave=False):
            lt_estimate = gross_ss.live_time.values[i]
            if i == 0 or bg_ss.n_samples > 1:
                bg_spectrum = bg_ss.spectra.values[i]
                valid_bins = self._calculate_valid_bins(bg_spectrum)
                seed_spectra = self._seeds.values[:, valid_bins]
                bg_cps = bg_ss.total_counts.values[i] / bg_ss.live_time.values[i]
            estimated_bg_counts = lt_estimate * bg_cps
            probas[i, :] = self._predict_single(
                seed_spectra,
                gross_ss.spectra.values[i][valid_bins],
                bg_spectrum[valid_bins],
                estimated_bg_counts,
                normalize_scores=normalize_scores
            )
        max_indices = probas.argmax(axis=1)
        gross_ss.predictions = self._classes[max_indices]
        gross_ss.prediction_probas = pd.DataFrame(probas, columns=self._classes)

    def save(self, file_path: str):
        """Saves the model to a file.

        Args:
            file_path: a string representing the file path at which to save the model.

        Returns:
            None

        Raises:
            ValueError: Raised when the given file path already exists.
        """
        if os.path.exists(file_path):
            raise ValueError("Path already exists.")
        self._seeds.to_hdf(file_path, "seeds")
        pd.DataFrame(self._classes).to_hdf(file_path, "classes")

    def load(self, file_path: str):
        """Loads the model from a file.

        Args:
            file_path: a string representing the file path from which to load the model.

        Returns:
            None

        Raises:
            None
        """
        self._seeds = pd.read_hdf(file_path, "seeds")
        self._classes = pd.read_hdf(file_path, "classes").values.flatten()
        self._n_bins = self._seeds.shape[1]

    def serialize(self) -> bytes:
        """Converts the model to a bytes object.

        Args:
            None

        Returns:
            Returns a bytes object representing the binary of an HDF file.

        Raises:
            None
        """
        self.save(self._temp_file_path)
        with open(self._temp_file_path, "rb") as f:
            data = f.read()
        os.remove(self._temp_file_path)
        return data

    def deserialize(self, stream: bytes):
        """Populates the current model with the give bytes object.

        Args:
            stream: a bytes object containing the model information.

        Returns:
            None

        Raises:
            None
        """
        with open(self._temp_file_path, "wb") as f:
            f.write(stream)
        self.load(self._temp_file_path)
        os.remove(self._temp_file_path)


class ZeroTotalCountsError(ValueError):
    pass


class NegativeSpectrumError(ValueError):
    pass
