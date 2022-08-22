# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains a Poisson-Bayes classifier."""
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from riid.data import SampleSet
from riid.models import TFModelBase
from scipy.stats import poisson


class PoissonBayes(TFModelBase):
    """This class implements a Bayes classifier for poisson distributed data."""

    def __init__(self, seeds_ss: SampleSet = None, normalize_scores: bool = False):
        """Initializes the classifier with seeds.

        This constructor takes "seed" spectra as input to be used as the shapes against which
        classifications will be performed. It will apply a PMF to the SampleSet.

        Args:
            seeds: an optional SampleSet of seed spectra for each potential source of interest.
                A background seed may be included if classification as background is desired.

        Raises:
            NegativeSpectrumError: raised if any seed spectrum has negative counts in any bin.
            ZeroTotalCountsError: raised if any seed spectrum contains zero total counts.

        """
        super().__init__()

        if seeds_ss is None:
            return

        if seeds_ss.n_samples <= 0:
            raise ValueError("Argument 'seeds_ss' must contain at least one seed.")
        if (seeds_ss.spectra.values < 0).any():
            msg = "Argument 'seeds_ss' can't contain any spectra with negative values."
            raise NegativeSpectrumError(msg)
        if (seeds_ss.spectra.values.sum(axis=1) <= 0).any():
            msg = "Argument 'seeds_ss' can't contain any spectra with zero total counts."
            raise ZeroTotalCountsError(msg)

        seeds_ss.normalize(p=1)
        self._seeds = seeds_ss.spectra
        self._n_channels = seeds_ss.n_channels
        self.model = None
        self.normalize_scores = normalize_scores
        self._proba_columns = seeds_ss.sources.columns
        self._labels = seeds_ss.get_labels(target_level="Seed")
        self._make_model()
        self.initialize_info()

    def _make_model(self):
        """Makes tensorflow model implementation of PoissonBayes."""
        spectrum_input = tf.keras.layers.Input(
            shape=(self._n_channels,),
            name="spectrum"
        )
        total_counts = tf.math.reduce_sum(
            spectrum_input,
            axis=1,
            name="total_counts"
        )

        fg_pmfs = tf.constant(self._seeds.values)

        expected_spectra = tf.math.multiply(
            tf.cast(fg_pmfs, tf.float32),
            tf.expand_dims(tf.expand_dims(total_counts, axis=-1), axis=-1)
        )

        if self.normalize_scores:
            expected_spectra = tf.concat(
                [expected_spectra, tf.expand_dims(spectrum_input, axis=1)],
                axis=1
            )

        reshaped_spectrum_input = tf.expand_dims(spectrum_input, axis=1)

        P = tfp.distributions.Poisson(
            expected_spectra,
            force_probs_to_zero_outside_support=True,
            allow_nan_stats=False,
        )
        all_probas = P.log_prob(reshaped_spectrum_input)
        prediction_probas = tf.math.reduce_sum(all_probas, axis=2)

        if self.normalize_scores:
            n_labels = len(self._labels)
            best_proba = tf.gather(prediction_probas, [n_labels], axis=1)
            prediction_probas = -prediction_probas / best_proba
            prediction_probas = tf.gather(prediction_probas, np.arange(n_labels), axis=1)
            prediction_indices = tf.math.argmax(prediction_probas, axis=1)
        else:
            # Determine max index for predictions
            prediction_indices = tf.math.argmax(prediction_probas, axis=1)

        # Build and compile the model
        model = tf.keras.Model(
            [spectrum_input],
            [prediction_indices, prediction_probas]
        )
        model.compile()
        self.model = model

    def _calculate_valid_channels(self, bg_spectrum: np.ndarray) -> np.ndarray:
        """Calculates which bins are valid for use in classification based on the values present in
        both the seeds and the given background spectrum.

        Args:
            bg_spectrum: Defines a numpy.ndarray representing a single background spectrum,
                with the shape: (n_channels,).

        Returns:
            A numpy.ndarray of booleans representing which channels are valid (True) and which
            are not (False).

        """
        return ((self._seeds.values.T + bg_spectrum[:, None]) > 0).prod(axis=1).astype(bool)

    def predict(self, ss: SampleSet):
        """Attempts to classify the provided spectra using the seeds with which the
        PoissonBayes object was originally initialized.

        Args:
            ss: Defines a SampleSet of `n` spectra where `n` >= 1.

        Raises:
            ValueError: Raised when no spectra are provided.
            ValueError: Raised when spectrum channel sizes are inconsisent.

        """
        if ss.n_samples <= 0:
            raise ValueError("No spectr[a|um] provided!")
        if ss.n_channels != self._n_channels:
            msg = "The provided spectra must have the same number of channels same as the seeds!  "
            msg += "Seed spectra contain {} channels.".format(self._n_channels)
            raise ValueError(msg)

        _, probas = self.model.predict([ss.spectra.values])
        ss.prediction_probas = pd.DataFrame(
            probas,
            columns=self._proba_columns
        )

    def _predict_single(self, seed_spectra: np.ndarray, bg_spectrum: np.ndarray, bg_cps: float,
                        event_spectrum: np.ndarray, event_live_time: float,
                        normalize_scores: bool = False) -> np.ndarray:
        total_counts = event_spectrum.sum()
        event_spectrum = event_spectrum.round().astype(int)
        expected_bg_counts = bg_cps * event_live_time
        expected_bg_spectrum = (bg_spectrum * expected_bg_counts).astype(int)
        expected_fg_counts = total_counts - expected_bg_counts
        expected_fg_spectra = (expected_fg_counts * seed_spectra).clip(1).astype(int)
        expected_spectra = expected_bg_spectrum + expected_fg_spectra

        if normalize_scores:
            expected_spectra = np.vstack(
                (
                    expected_spectra,
                    event_spectrum,
                )
            )

        probas = poisson\
            .logpmf(event_spectrum, expected_spectra)\
            .sum(axis=1)

        if normalize_scores:
            probas = probas[:-1] - probas[-1]

        return probas

    def predict_old(self, ss: SampleSet, bg_ss: SampleSet, normalize_scores: bool = False,
                    verbose: bool = False):

        # pd.DataFrame(fg_seeds_ss.spectra.values + bg_ss.spectra.values)
        bg_spectrum = bg_ss.spectra.iloc[0].values
        bg_cps = bg_ss.info.iloc[0].gross_counts / bg_ss.info.iloc[0].live_time

        probas = np.zeros([ss.n_samples, len(self._labels)])
        for i in range(ss.n_samples):
            probas[i, :] = self._predict_single(
                self._seeds.values,
                bg_spectrum,
                bg_cps,
                ss.spectra.values[i],
                ss.info.live_time[i],
                normalize_scores=normalize_scores
            )
            if verbose:
                percent_complete = 100 * i / ss.n_samples
                logging.info(f"{percent_complete:.0f}% complete")
        ss.prediction_probas = pd.DataFrame(
            probas,
            columns=self._proba_columns
        )

    def to_tflite(self, file_path: str = None, quantize: bool = False):
        """Exports model to TFLite file.

        Args:
            file_path:
            quantize:

        Returns:
            None.

        Raises:
            NotImplementedError:  Raised when function is called; this function has not yet
                been implemented.
        """
        raise NotImplementedError(
            "TFLite conversion is not supported for TFP layers at this time (TensorFlow 2.4.0)"
        )


class ZeroTotalCountsError(ValueError):
    """An exception that indicates that a total count of zero has
    been found, which means the model statistics cannot be calculated."""
    pass


class NegativeSpectrumError(ValueError):
    """An exception that indicates that a negative spectrum value has
    been found, which means that the model statistics cannot be calculated."""
    pass
