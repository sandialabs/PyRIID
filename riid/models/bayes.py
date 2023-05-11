# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains a Poisson-Bayes classifier."""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from riid.data import SampleSet
from riid.models import TFModelBase


class PoissonBayesClassifier(TFModelBase):
    def __init__(self):
        """This Poisson-Bayes classifier calculates the conditional Poisson log
        probability of each seed spectrum given the measurement.

        This implementation is an adaptation of a naive Bayes classifier, a formal
        description of which can be found in ESLII:
            Hastie, Trevor, et al. The elements of statistical learning: data mining,
                inference, and prediction. Vol. 2. New York: springer, 2009.
        For this model, each channel (read: feature) is treated as a Poisson random
        variable and expectations are provided by the user in the form of
        seeds rather than learned. Like the model described in ESLII, all classes are
        considered equally likely and features are assumed to be independent.
        The latter assumption is generally not true, however it makes estimation far
        more simple.
        """
        super().__init__()

    def _create_model(self, seeds_ss: SampleSet):
        self._seeds = tf.constant(tf.convert_to_tensor(
            seeds_ss.spectra.values,
            dtype=tf.float32
        ))

        # Inputs
        gross_spectrum_input = tf.keras.layers.Input(
            shape=self.seeds_ss.n_channels,
            name="gross_spectrum"
        )
        gross_live_time_input = tf.keras.layers.Input(
            shape=(),
            name="gross_live_time"
        )
        bg_spectrum_input = tf.keras.layers.Input(
            shape=self.seeds_ss.n_channels,
            name="bg_spectrum"
        )
        bg_live_time_input = tf.keras.layers.Input(
            shape=(),
            name="bg_live_time"
        )

        # Compute expected_seed_spectrums
        gross_total_counts = tf.reduce_sum(gross_spectrum_input, axis=1)
        bg_total_counts = tf.reduce_sum(bg_spectrum_input, axis=1)
        bg_count_rate = tf.divide(bg_total_counts, bg_live_time_input)
        expected_bg_counts = tf.multiply(bg_count_rate, gross_live_time_input)
        expected_fg_counts = tf.subtract(gross_total_counts, expected_bg_counts)
        normalized_bg_spectrum = tf.divide(
            bg_spectrum_input,
            tf.expand_dims(bg_total_counts, axis=1)
        )
        expected_bg_spectrum = tf.multiply(
            normalized_bg_spectrum,
            tf.expand_dims(expected_bg_counts, axis=1)
        )
        expected_fg_spectrum = tf.multiply(
            self._seeds,
            tf.expand_dims(tf.expand_dims(
                expected_fg_counts,
                axis=-1
            ), axis=-1)
        )
        max_value = tf.math.reduce_max(expected_fg_spectrum)
        expected_fg_spectrum = tf.clip_by_value(expected_fg_spectrum, 1e-8, max_value)
        expected_gross_spectum = tf.add(
            expected_fg_spectrum,
            tf.expand_dims(expected_bg_spectrum, axis=1)
        )

        poisson_dist = tfp.distributions.Poisson(expected_gross_spectum)
        all_probas = poisson_dist.log_prob(
            tf.expand_dims(gross_spectrum_input, axis=1)
        )
        prediction_probas = tf.math.reduce_sum(all_probas, axis=2)

        model_inputs = (
            gross_spectrum_input,
            gross_live_time_input,
            bg_spectrum_input,
            bg_live_time_input,
        )
        self.model = tf.keras.Model(model_inputs, prediction_probas)
        self.model.compile()

    def fit(self, seeds_ss: SampleSet = None):
        """Constructs a TF-based implementation of a poisson-bayes classifier in terms
        of the given seeds.

        Args:
            seeds_ss (SampleSet): Defines a SampleSet of `n` foreground
                seed spectra where `n` >= 1.

        Raises:
            ValueError: Raised when no seeds are provided.
            NegativeSpectrumError: raised if any seed spectrum has negative counts in any bin.
            ZeroTotalCountsError: raised if any seed spectrum contains zero total counts.
        """
        if seeds_ss.n_samples <= 0:
            raise ValueError("Argument 'seeds_ss' must contain at least one seed.")
        if (seeds_ss.spectra.values < 0).any():
            msg = "Argument 'seeds_ss' can't contain any spectra with negative values."
            raise NegativeSpectrumError(msg)
        if (seeds_ss.spectra.values.sum(axis=1) <= 0).any():
            msg = "Argument 'seeds_ss' can't contain any spectra with zero total counts."
            raise ZeroTotalCountsError(msg)

        self.seeds_ss = seeds_ss
        self._create_model(self.seeds_ss)

    def predict(self, gross_ss: SampleSet = None, bg_ss: SampleSet = None,
                normalize_scores: bool = False, verbose: bool = False):
        """Uses the Poisson-Bayes model to output prediction
        probabilities for every spectra in gross_ss SampleSet
        (and optional background Sample Set)

        Args:
            gross_ss (SampleSet): Defines a SampleSet of `n` gross spectra where `n` >= 1.
            bg_ss (SampleSet): Defines a SampleSet of `n` background spectra where `n` >= 1.
            normalize_scores (bool): normalize prediction probabilities.
                When True, this makes the probabilities positive and rescales them
                by the minimum value present in given the dataset.
                While this can be helpful in terms of visualizing probabilities in log scale,
                it can adversely affects one's ability to detect significantly anomalous signatures.
        """
        gross_spectra = tf.convert_to_tensor(gross_ss.spectra.values, dtype=tf.float32)
        gross_lts = tf.convert_to_tensor(gross_ss.info.live_time.values, dtype=tf.float32)
        bg_spectra = tf.convert_to_tensor(bg_ss.spectra.values, dtype=tf.float32)
        bg_lts = tf.convert_to_tensor(bg_ss.info.live_time.values, dtype=tf.float32)

        prediction_probas = self.model.predict((
            gross_spectra, gross_lts, bg_spectra, bg_lts
        ), batch_size=512, verbose=verbose)

        # Normalization
        if normalize_scores:
            rows_min = np.min(prediction_probas, axis=1)
            prediction_probas = prediction_probas - rows_min[:, np.newaxis]

        gross_ss.prediction_probas = pd.DataFrame(
            prediction_probas,
            columns=self.seeds_ss.sources.columns
        )


class ZeroTotalCountsError(ValueError):
    """An exception that indicates that a total count of zero has
    been found, which means the model statistics cannot be calculated."""
    pass


class NegativeSpectrumError(ValueError):
    """An exception that indicates that a negative spectrum value has
    been found, which means that the model statistics cannot be calculated."""
    pass
