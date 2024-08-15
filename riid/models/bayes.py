# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains the Poisson-Bayes classifier."""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.layers import Add, Input, Multiply, Subtract
from keras.api.models import Model

from riid import SampleSet
from riid.models.base import PyRIIDModel
from riid.models.layers import (ClipByValueLayer, DivideLayer, ExpandDimsLayer,
                                PoissonLogProbabilityLayer, ReduceMaxLayer,
                                ReduceSumLayer, SeedLayer)


class PoissonBayesClassifier(PyRIIDModel):
    """Classifier calculating the conditional Poisson log probability of each seed spectrum
    given the measurement.

    This implementation is an adaptation of a naive Bayes classifier, a formal description of
    which can be found in ESLII:

    Hastie, Trevor, et al. The elements of statistical learning: data mining, inference, and
    prediction. Vol. 2. New York. Springer, 2009.

    For this model, each spectrum channel is treated as a Poisson random variable and
    expectations are provided by the user in the form of seeds rather than learned.
    Like the model described in ESLII, all classes are considered equally likely and features
    are assumed to be conditionally independent.
    """
    def __init__(self):
        super().__init__()

        self._update_custom_objects("ReduceSumLayer", ReduceSumLayer)
        self._update_custom_objects("ReduceMaxLayer", ReduceMaxLayer)
        self._update_custom_objects("DivideLayer", DivideLayer)
        self._update_custom_objects("ExpandDimsLayer", ExpandDimsLayer)
        self._update_custom_objects("ClipByValueLayer", ClipByValueLayer)
        self._update_custom_objects("PoissonLogProbabilityLayer", PoissonLogProbabilityLayer)
        self._update_custom_objects("SeedLayer", SeedLayer)

    def fit(self, seeds_ss: SampleSet):
        """Construct a TF-based implementation of a poisson-bayes classifier in terms
        of the given seeds.

        Args:
            seeds_ss: `SampleSet` of `n` foreground seed spectra where `n` >= 1.

        Raises:
            - `ValueError` when no seeds are provided
            - `NegativeSpectrumError` when any seed spectrum has negative counts in any bin
            - `ZeroTotalCountsError` when any seed spectrum contains zero total counts
        """
        if seeds_ss.n_samples <= 0:
            raise ValueError("Argument 'seeds_ss' must contain at least one seed.")
        if (seeds_ss.spectra.values < 0).any():
            msg = "Argument 'seeds_ss' can't contain any spectra with negative values."
            raise NegativeSpectrumError(msg)
        if (seeds_ss.spectra.values.sum(axis=1) <= 0).any():
            msg = "Argument 'seeds_ss' can't contain any spectra with zero total counts."
            raise ZeroTotalCountsError(msg)

        self._seeds = tf.convert_to_tensor(
            seeds_ss.spectra.values,
            dtype=tf.float32
        )

        # Inputs
        gross_spectrum_input = Input(shape=(seeds_ss.n_channels,),
                                     name="gross_spectrum")
        gross_live_time_input = Input(shape=(),
                                      name="gross_live_time")
        bg_spectrum_input = Input(shape=(seeds_ss.n_channels,),
                                  name="bg_spectrum")
        bg_live_time_input = Input(shape=(),
                                   name="bg_live_time")
        model_inputs = (
            gross_spectrum_input,
            gross_live_time_input,
            bg_spectrum_input,
            bg_live_time_input,
        )

        # Input statistics
        gross_total_counts = ReduceSumLayer(name="gross_total_counts")(gross_spectrum_input, axis=1)
        bg_total_counts = ReduceSumLayer(name="bg_total_counts")(bg_spectrum_input, axis=1)
        bg_count_rate = DivideLayer(name="bg_count_rate")([bg_total_counts, bg_live_time_input])

        gross_spectrum_input_expanded = ExpandDimsLayer(
            name="gross_spectrum_input_expanded"
        )(gross_spectrum_input, axis=1)
        bg_total_counts_expanded = ExpandDimsLayer(
            name="bg_total_counts_expanded"
        )(bg_total_counts, axis=1)

        # Expectations
        seed_layer = SeedLayer(self._seeds)(model_inputs)
        seed_layer_expanded = ExpandDimsLayer()(seed_layer, axis=0)
        expected_bg_counts = Multiply(
            trainable=False,
            name="expected_bg_counts"
        )([bg_count_rate, gross_live_time_input])
        expected_bg_counts_expanded = ExpandDimsLayer(
            name="expected_bg_counts_expanded"
        )(expected_bg_counts, axis=1)
        normalized_bg_spectrum = DivideLayer(
            name="normalized_bg_spectrum"
        )([bg_spectrum_input, bg_total_counts_expanded])
        expected_bg_spectrum = Multiply(
            trainable=False,
            name="expected_bg_spectrum"
        )([normalized_bg_spectrum, expected_bg_counts_expanded])
        expected_fg_counts = Subtract(
            trainable=False,
            name="expected_fg_counts"
        )([gross_total_counts, expected_bg_counts])
        expected_fg_counts_expanded = ExpandDimsLayer(
            name="expected_fg_counts_expanded"
        )(expected_fg_counts, axis=-1)
        expected_fg_counts_expanded2 = ExpandDimsLayer(
            name="expected_fg_counts_expanded2"
        )(expected_fg_counts_expanded, axis=-1)
        expected_fg_spectrum = Multiply(
            trainable=False,
            name="expected_fg_spectrum"
        )([seed_layer_expanded, expected_fg_counts_expanded2])
        max_fg_value = ReduceMaxLayer(
            name="max_fg_value"
        )(expected_fg_spectrum)
        expected_fg_spectrum = ClipByValueLayer(
            name="clip_expected_fg_spectrum"
        )(expected_fg_spectrum, clip_value_min=1e-8, clip_value_max=max_fg_value)
        expected_bg_spectrum_expanded = ExpandDimsLayer(
            name="expected_bg_spectrum_expanded"
        )(expected_bg_spectrum, axis=1)
        expected_gross_spectrum = Add(
            trainable=False,
            name="expected_gross_spectrum"
        )([expected_fg_spectrum, expected_bg_spectrum_expanded])

        # Compute probabilities
        log_probabilities = PoissonLogProbabilityLayer(
            name="log_probabilities"
        )([expected_gross_spectrum, gross_spectrum_input_expanded])
        summed_log_probabilities = ReduceSumLayer(
            name="summed_log_probabilities"
        )(log_probabilities, axis=2)

        # Assemble model
        self.model = Model(model_inputs, summed_log_probabilities)
        self.model.compile()

        self.target_level = "Seed"
        sources_df = seeds_ss.sources.T.groupby(self.target_level, sort=False).sum().T
        self.model_outputs = sources_df.columns.values.tolist()

    def predict(self, gross_ss: SampleSet, bg_ss: SampleSet,
                normalize_scores: bool = False, verbose: bool = False):
        """Compute the conditional Poisson log probability between spectra in a `SampleSet` and
        the seeds to which the model was fit.

        Args:
            gross_ss: `SampleSet` of `n` gross spectra where `n` >= 1
            bg_ss: `SampleSet` of `n` background spectra where `n` >= 1
            normalize_scores (bool): whether to normalize prediction probabilities
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
            columns=pd.MultiIndex.from_tuples(
                self.get_model_outputs_as_label_tuples(),
                names=SampleSet.SOURCES_MULTI_INDEX_NAMES
            )
        )


class ZeroTotalCountsError(ValueError):
    """All spectrum channels are zero."""
    pass


class NegativeSpectrumError(ValueError):
    """At least one spectrum channel is negative."""
    pass
