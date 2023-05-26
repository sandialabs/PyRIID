# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains a multi-layer perceptron classifier."""
import copy
import json
import os
from typing import Any, List, Tuple

import numpy as np
import onnxruntime
import pandas as pd
import tensorflow as tf
import tf2onnx
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tf2onnx import logging
from tqdm import tqdm

from riid.data import SampleSet
from riid.models import ModelInput, TFModelBase
from riid.models.losses import (build_semisupervised_loss_func,
                                normal_nll_diff, poisson_nll_diff,
                                reconstruction_error, sse_diff,
                                weighted_sse_diff, sparsemax,
                                SparsemaxLoss)
from riid.models.metrics import RunningAverage, multi_f1, single_f1

logging.basicConfig(level=logging.WARNING)


def _l1_norm(x):
    sums = tf.reduce_sum(x, axis=-1)
    l1_norm = x / tf.reshape(sums, (-1, 1))
    return l1_norm


def _get_reordered_spectra(old_spectra_df: pd.DataFrame, old_sources_df: pd.DataFrame,
                           new_sources_columns, target_level) -> pd.DataFrame:
    collapsed_sources_df = old_sources_df\
        .groupby(axis=1, level=target_level)\
        .sum()
    reordered_spectra_df = old_spectra_df.iloc[
        collapsed_sources_df[
            new_sources_columns
        ].idxmax()
    ].reset_index(drop=True)

    return reordered_spectra_df


class MLPClassifier(TFModelBase):
    def __init__(self, hidden_layers: tuple = (512,), activation: str = "relu",
                 loss: str = "categorical_crossentropy",
                 optimizer: Any = Adam(learning_rate=0.01, clipnorm=0.001),
                 metrics: tuple = ("accuracy", "categorical_crossentropy", multi_f1, single_f1),
                 l2_alpha: float = 1e-4, activity_regularizer=l1(0), dropout: float = 0.0,
                 learning_rate: float = 0.01):
        """Initializes the classifier.

        The model is implemented as a tf.keras.Sequential object.

        Args:
            hidden_layers: Defines a tuple defining the number and size of dense layers.
            activation: Defines the activate function to use for each dense layer.
            loss: Defines the loss function to use for training.
            optimizer: Defines the tensorflow optimizer or optimizer name to use for training.
            metrics: Defines a list of metrics to be evaluating during training.
            l2_alpha: Defines the alpha value for the L2 regularization of each dense layer.
            activity_regularizer: Defines the regularizer function applied each dense layer output.
            dropout: Defines the amount of dropout to apply to each dense layer.
            learning_rate: the learning rate to use for an Adam optimizer.
        """
        super().__init__()

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.loss = loss
        if optimizer == "adam":
            self.optimizer = Adam(learning_rate=learning_rate)
        else:
            self.optimizer = optimizer
        self.optimizer = optimizer
        self.metrics = metrics
        self.l2_alpha = l2_alpha
        self.activity_regularizer = activity_regularizer
        self.dropout = dropout
        self.model = None

    def fit(self, ss: SampleSet, bg_ss: SampleSet = None,
            ss_input_type: ModelInput = ModelInput.GrossSpectrum,
            bg_ss_input_type: ModelInput = ModelInput.BackgroundSpectrum,
            batch_size: int = 200, epochs: int = 20,
            validation_split: float = 0.2, callbacks=None, val_ss: SampleSet = None,
            val_bg_ss: SampleSet = None, patience: int = 15, es_monitor: str = "val_loss",
            es_mode: str = "min", es_verbose=0, target_level="Isotope", verbose: bool = False):
        """Fits a model to the given SampleSet(s).

        Args:
            ss: Defines a SampleSet of `n` spectra where `n` >= 1 and the spectra are either
                foreground (AKA, "net") or gross.
            bg_ss: Defines a SampleSet of `n` spectra where `n` >= 1 and the spectra are background.
            batch_size: Defines the number of samples per gradient update.
            epochs: Defines maximum number of training iterations.
            validation_split: Defines the percentage of the training data to use as validation data.
            callbacks: Defines a list of callbacks to be passed to TensorFlow Model.fit() method.
            val_ss: Defines an optionally-provided provided validation set to be used instead of
                taking a portion of `ss` for validation.
            val_bg_ss: Defines an optionally-provided validation set to be used as background for
                val_ss.
            patience: Defines the number of epochs to wait for tf.keras.callbacks.EarlyStopping
                object.
            es_monitor: Defines the quantity to be monitored for tf.keras.callbacks.EarlyStopping
                object.
            es_mode: mode for tf.keras.callbacks.EarlyStopping object.
            es_verbose: Determines verbosity level for tf.keras.callbacks.EarlyStopping object.
            target_level: The source level to target for model output.
            verbose: Determines whether or not model training output is printed to the terminal.

        Returns:
            A tf.History object.

        Raises:
            ValueError: Raised when no spectra are provided as `ss` input.
        """
        if ss.n_samples <= 0:
            raise ValueError("No spectr[a|um] provided!")

        x_train = ss.get_samples().astype(float)
        source_contributions_df = ss.get_source_contributions(target_level=target_level)
        y_train = source_contributions_df.values.astype(float)
        if bg_ss:
            x_bg_train = bg_ss.get_samples().astype(float)

        if val_ss:
            if val_bg_ss:
                val_data = (
                    [val_ss.get_samples().astype(float), val_bg_ss.get_samples().astype(float)],
                    val_ss.get_source_contributions().values.astype(float),
                )
            else:
                val_data = (
                    val_ss.get_samples().astype(float),
                    val_ss.get_source_contributions().values.astype(float),
                )
            validation_split = None
        else:
            val_data = None
            row_order = np.arange(x_train.shape[0])
            np.random.shuffle(row_order)
            # Enforce random validation split through shuffling
            x_train = x_train[row_order]
            y_train = y_train[row_order]

            if bg_ss:
                x_bg_train = x_bg_train[row_order]

        if not self.model:
            spectra_input = tf.keras.layers.Input(
                x_train.shape[1],
                name=ss_input_type.name
            )
            inputs = [spectra_input]

            if bg_ss:
                background_spectra_input = tf.keras.layers.Input(
                    x_bg_train.shape[1],
                    name=bg_ss_input_type.name
                )
                inputs.append(background_spectra_input)

            if len(inputs) > 1:
                x = tf.keras.layers.Concatenate()(inputs)
            else:
                x = inputs[0]

            for layer, nodes in enumerate(self.hidden_layers):
                if layer == 0:
                    x = Dense(
                        nodes,
                        activation=self.activation,
                        activity_regularizer=self.activity_regularizer,
                        kernel_regularizer=l2(self.l2_alpha),
                    )(x)
                else:
                    x = Dense(
                        nodes,
                        activation=self.activation,
                        activity_regularizer=self.activity_regularizer,
                        kernel_regularizer=l2(self.l2_alpha),
                    )(x)
                if self.dropout > 0:
                    x = Dropout(self.dropout)(x)

            output = Dense(y_train.shape[1], activation="softmax")(x)
            self.model = tf.keras.models.Model(inputs, output)
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        es = EarlyStopping(
            monitor=es_monitor,
            patience=patience,
            verbose=es_verbose,
            restore_best_weights=True,
            mode=es_mode,
        )

        if callbacks:
            callbacks.append(es)
        else:
            callbacks = [es]

        if bg_ss:
            X_data = [x_train, x_bg_train]
            self.model_inputs = (ss_input_type, bg_ss_input_type)
        else:
            X_data = x_train
            self.model_inputs = (ss_input_type,)

        history = self.model.fit(
            X_data,
            y_train,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split,
            validation_data=val_data,
            callbacks=callbacks,
            shuffle=True,
            batch_size=batch_size,
        )

        # Initialize model information
        self.target_level = target_level
        self.model_outputs = source_contributions_df.columns.values
        self.initialize_info()
        # TODO: get rid of the following line in favor of a normalization layer
        self._info["normalization"] = ss.spectra_state

        return history

    def predict(self, ss: SampleSet, bg_ss: SampleSet = None):
        """Classifies the spectra in the provided SampleSet(s).

        Results are stored inside the first SampleSet's prediction-related properties.

        Args:
            ss: Defines a SampleSet of `n` spectra where `n` >= 1 and the spectra are either
                foreground (AKA, "net") or gross.
            bg_ss: Defines a SampleSet of `n` spectra where `n` >= 1 and the spectra are background.

        """
        x_test = ss.get_samples().astype(float)
        if bg_ss:
            X = [x_test, bg_ss.get_samples().astype(float)]
        else:
            X = x_test
        results = self.model.predict(X)  # output size will be n_samples by n_labels

        col_level_idx = SampleSet.SOURCES_MULTI_INDEX_NAMES.index(self.target_level)
        col_level_subset = SampleSet.SOURCES_MULTI_INDEX_NAMES[:col_level_idx+1]
        ss.prediction_probas = pd.DataFrame(
            data=results,
            columns=pd.MultiIndex.from_tuples(
               self.model_outputs, names=col_level_subset
            )
        )

        ss.classified_by = self.info["model_id"]


class MultiEventClassifier(TFModelBase):
    def __init__(self, hidden_layers: tuple = (512,), activation: str = "relu",
                 loss: str = "categorical_crossentropy",
                 optimizer: Any = Adam(learning_rate=0.01, clipnorm=0.001),
                 metrics: list = ["accuracy", "categorical_crossentropy", multi_f1, single_f1],
                 l2_alpha: float = 1e-4, activity_regularizer: tf.keras.regularizers = l1(0),
                 dropout: float = 0.0, learning_rate: float = 0.01):
        """Initializes the classifier.
        The model is implemented as a tf.keras.Model object.

        Args:
            hidden_layers: Defines a tuple containing the number and size of dense layers.
            activation: Defines the activate function to use for each dense layer.
            loss: Defines the string name of the loss function to use for training.
            optimizer: Defines the string name of the optimizer to use for training.
            metrics: Defines a list of metrics to be evaluating during training.
            l2_alpha: Defines the alpha value for the L2 regularization of each dense layer.
            activity_regularizer: Defines the regularizer function applied each dense layer output.
            dropout: Defines the amount of dropout to apply to each dense layer.
            learning_rate: the learning rate to use for an Adam optimizer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.loss = loss
        if optimizer == "adam":
            self.optimizer = Adam(learning_rate=learning_rate)
        else:
            self.optimizer = optimizer
        self.metrics = metrics
        self.l2_alpha = l2_alpha
        self.activity_regularizer = activity_regularizer
        self.dropout = dropout
        self.model = None

    def fit(self, list_of_ss: List[SampleSet], target_contributions: pd.DataFrame,
            batch_size: int = 200, epochs: int = 20,
            validation_split: float = 0.2, callbacks: list = None,
            val_model_ss_list: SampleSet = None,
            val_model_target_contributions: pd.DataFrame = None,
            patience: int = 15, es_monitor: str = "val_loss", es_mode: str = "min",
            es_verbose: bool = False, target_level="Isotope", verbose: bool = False):
        """Fits a model to the given SampleSet(s).

        Args:
            list_of_ss: Defines a list of SampleSets which have prediction_probas populated from
                single-event classifiers.
            target_contributions: Defines a DataFrame of the contributions for each
                observation. Column titles are the desired label strings.
            batch_size: Defines the number of samples per gradient update.
            epochs: Defines the maximum number of training iterations.
            validation_split: Defines the percentage of the training data to use as validation data.
            callbacks: Defines a list of callbacks to be passed to TensorFlow Model.fit() method.
            val_model_ss_list: Defines an optionally-provided validation set to be used instead of
                taking a portion of `ss` for validation.
            val_model_target_contributions: Defines the target contributions to the model for
                each sample.
            patience: Defines the number of epochs to wait for tf.keras.callbacks.EarlyStopping
                object.
            es_monitor: Defintes the quantity to be monitored for tf.keras.callbacks.EarlyStopping
                object.
            es_mode: Defines the mode for tf.keras.callbacks.EarlyStopping object.
            es_verbose: Determines the verbosity level for tf.keras.callbacks.EarlyStopping object.
            target_level: The source level to target for model output.
            verbose: Determines whether or not the training output is printed to the terminal.

        Returns:
            A tf.History object.

        Raises:
            ValueError: Raised when no predictions are provided with `list_of_ss` input.
        """
        if len(list_of_ss) <= 0:
            raise ValueError("No model predictions provided!")

        x_train = [ss.prediction_probas.values for ss in list_of_ss]
        y_train = target_contributions.values

        if val_model_ss_list and val_model_target_contributions:
            val_data = (
                    [ss.prediction_probas.values for ss in val_model_ss_list],
                    val_model_target_contributions.values,
                )
            validation_split = None
        else:
            val_data = None
            row_order = np.arange(x_train[0].shape[0])
            np.random.shuffle(row_order)
            # Enforce random validation split through shuffling
            x_train = [i[row_order] for i in x_train]
            y_train = y_train[row_order]

        if not self.model:
            inputs = []
            for ss in list_of_ss:
                input_from_single_event_model = tf.keras.layers.Input(
                    ss.prediction_probas.shape[1],
                    name=ss.classified_by
                )
                inputs.append(input_from_single_event_model)

            if len(inputs) > 1:
                x = tf.keras.layers.Concatenate()(inputs)
            else:
                x = inputs[0]

            for layer, nodes in enumerate(self.hidden_layers):
                if layer == 0:
                    x = Dense(
                            nodes,
                            activation=self.activation,
                            activity_regularizer=self.activity_regularizer,
                            kernel_regularizer=l2(self.l2_alpha),
                        )(x)
                else:
                    x = Dense(
                            nodes,
                            activation=self.activation,
                            activity_regularizer=self.activity_regularizer,
                            kernel_regularizer=l2(self.l2_alpha),
                        )(x)
                if self.dropout > 0:
                    x = Dropout(self.dropout)(x)

            output = Dense(y_train.shape[1], activation="softmax")(x)
            self.model = tf.keras.models.Model(inputs, output)
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        es = EarlyStopping(
            monitor=es_monitor,
            patience=patience,
            verbose=es_verbose,
            restore_best_weights=True,
            mode=es_mode,
        )

        if callbacks:
            callbacks.append(es)
        else:
            callbacks = [es]

        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split,
            validation_data=val_data,
            callbacks=callbacks,
            shuffle=True,
            batch_size=batch_size,
        )

        # Initialize model info, update output/input information
        self.target_level = target_level
        self.model_outputs = target_contributions.columns.values
        self.initialize_info()
        self.info["model_inputs"] = tuple(
            [(ss.classified_by, ss.prediction_probas.shape[1]) for ss in list_of_ss]
        )

        return history

    def predict(self, list_of_ss: List[SampleSet]) -> pd.DataFrame:
        """Classifies the spectra in the provided SampleSet(s) based on each Sampleset's results.

        Args:
            list_of_ss: Defines a list of SampleSets which had predictions made by
                single-event models.

        Returns:
            A DataFrame of predicted results for the Sampleset(s).
        """
        X = [ss.prediction_probas for ss in list_of_ss]
        results = self.model.predict(X)  # output size will be n_samples by n_labels

        col_level_idx = SampleSet.SOURCES_MULTI_INDEX_NAMES.index(self.target_level)
        col_level_subset = SampleSet.SOURCES_MULTI_INDEX_NAMES[:col_level_idx+1]
        results_df = pd.DataFrame(
            data=results,
            columns=pd.MultiIndex.from_tuples(
               self.model_outputs, names=col_level_subset
            )
        )
        return results_df
    
class MLPClassifierWithGeneration(TFModelBase):
    def __init__(self, hidden_layers: tuple = (512,), activation: str = "relu",
                 loss: str = "categorical_crossentropy",
                 optimizer: Any = Adam(learning_rate=0.01, clipnorm=0.001),
                 metrics: tuple = ("accuracy", "categorical_crossentropy", multi_f1, single_f1),
                 l2_alpha: float = 1e-4, activity_regularizer=l1(0), dropout: float = 0.0,
                 learning_rate: float = 0.01):
        """Initializes the classifier.

        The model is implemented as a tf.keras.Sequential object.

        Args:
            hidden_layers: Defines a tuple defining the number and size of dense layers.
            activation: Defines the activate function to use for each dense layer.
            loss: Defines the loss function to use for training.
            optimizer: Defines the tensorflow optimizer or optimizer name to use for training.
            metrics: Defines a list of metrics to be evaluating during training.
            l2_alpha: Defines the alpha value for the L2 regularization of each dense layer.
            activity_regularizer: Defines the regularizer function applied each dense layer output.
            dropout: Defines the amount of dropout to apply to each dense layer.
            learning_rate: the learning rate to use for an Adam optimizer.
        """
        super().__init__()

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.loss = loss
        if optimizer == "adam":
            self.optimizer = Adam(learning_rate=learning_rate)
        else:
            self.optimizer = optimizer
        self.optimizer = optimizer
        self.metrics = metrics
        self.l2_alpha = l2_alpha
        self.activity_regularizer = activity_regularizer
        self.dropout = dropout
        self.model = None

    def fit(self,
            static_syn: StaticSynthesizer, fg_seeds_ss: SampleSet, bg_seeds_ss: SampleSet,
            ss_input_type: ModelInput = ModelInput.GrossSpectrum,
            batch_size: int = 200, epochs: int = 20,
            callbacks=None, val_gross_ss: SampleSet = None,
            patience: int = 15, es_monitor: str = "val_loss",
            es_mode: str = "min", es_verbose=0, target_level="Isotope", verbose: bool = False,
            generator_verbose: bool = False, tf_cache: bool = False):

        """ Fits a model over generate synthetically generated GROSS samples in real time.
            If StaticSynthesizer's random_state is set to None and tf_cache is set to False,
            every batch for every epoch will generate new data.
            If tf_cache is set to True, the first epoch will generate new data per batch.
            Subsequent epochs will use the same data as the first epoch.
            WARNING: If StaticSynthesizer's random state is NOT None, the training will
            only use the first batch's data for every subsequent batch.


        Args:
            static_syn: a StaticSynthesizer
            fg_seeds_ss: Contains spectra normalized by total counts to be
            used as the foreground (source only) component of spectra.
            bg_seeds_ss: Contains spectra normalized by total counts to be used
            ss_input_type: type of the sample set input
            batch_size: Defines the number of samples per gradient update.
            epochs: Defines maximum number of training iterations.
            callbacks: Defines a list of callbacks to be passed to TensorFlow Model.fit() method.
            val_gross_ss: Defines an optionally-provided provided GROSS validation set to be used.
            patience: Defines the number of epochs to wait for tf.keras.callbacks.EarlyStopping
                object.
            es_monitor: Defines the quantity to be monitored for tf.keras.callbacks.EarlyStopping
                object.
            es_mode: mode for tf.keras.callbacks.EarlyStopping object.
            es_verbose: Determines verbosity level for tf.keras.callbacks.EarlyStopping object.
            target_level: The source level to target for model output.
            verbose: Determines whether or not model training output is printed to the terminal.
            generator_verbose: Determines whether sample generation output is printed to terminal.
            tf_cache: Will repeat the first epochs accumilated sample for subsequent epochs.

        Returns:
            A tf.History object.

        """
        self.static_syn = static_syn

        if generator_verbose:
            total_num_of_samples = \
                self.static_syn.samples_per_seed * fg_seeds_ss.n_samples * bg_seeds_ss.n_samples
            print(f"samples_per_seeds: {self.static_syn.samples_per_seed}")
            print(f"batch_size: {batch_size}")
            print(f"epochs: {epochs}")
            print(f"total_num_samples:{total_num_of_samples}")

        # Initialize model information
        self.target_level = target_level

        spectrum_size = 512
        num_labels = 7
        if generator_verbose:
            print(f"spectrum_size: {spectrum_size}")
            print(f"num_labels: {num_labels}")

        # Tensorflow data pipeline dataset from generator
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: self.static_syn_batch_generator(
                batch_size,
                static_syn=static_syn,
                fg_seeds_ss=fg_seeds_ss,
                bg_seeds_ss=bg_seeds_ss,
                verbose=generator_verbose
            ),
            output_signature=(
                tf.TensorSpec(shape=(spectrum_size), dtype=tf.float32),
                tf.TensorSpec(shape=(num_labels), dtype=tf.float32),
            )
        )

        # If validation sample set is provided
        if val_gross_ss:
            val_data = (
                val_gross_ss.get_samples().astype(float),
                val_gross_ss.get_source_contributions().values.astype(float),
            )
        else:
            val_data = None

        if not self.model:
            spectra_input = tf.keras.layers.Input(
                spectrum_size,
                name=ss_input_type.name
            )
            inputs = [spectra_input]
            x = inputs[0]

            for layer, nodes in enumerate(self.hidden_layers):
                if layer == 0:
                    x = Dense(
                        nodes,
                        activation=self.activation,
                        activity_regularizer=self.activity_regularizer,
                        kernel_regularizer=l2(self.l2_alpha),
                    )(x)
                else:
                    x = Dense(
                        nodes,
                        activation=self.activation,
                        activity_regularizer=self.activity_regularizer,
                        kernel_regularizer=l2(self.l2_alpha),
                    )(x)
                if self.dropout > 0:
                    x = Dropout(self.dropout)(x)

            output = Dense(num_labels, activation="softmax")(x)
            self.model = tf.keras.models.Model(inputs, output)
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        es = EarlyStopping(
            monitor=es_monitor,
            patience=patience,
            verbose=es_verbose,
            restore_best_weights=True,
            mode=es_mode,
        )

        if callbacks:
            callbacks.append(es)
        else:
            callbacks = [es]

        if tf_cache:
            dataset_mode = dataset.batch(batch_size).cache().prefetch(buffer_size=1)
        else:
            dataset_mode = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        history = self.model.fit(
            dataset_mode,
            epochs=epochs,
            verbose=verbose,
            validation_data=val_data,
            callbacks=callbacks,
            shuffle=False,
            batch_size=batch_size,
        )

        self.initialize_info()

        return history

    def static_syn_batch_generator(
        self,
        batch_size: int, static_syn: StaticSynthesizer = None,
        fg_seeds_ss: SampleSet = None, bg_seeds_ss: SampleSet = None,
        target_level="Isotope", verbose: bool = False
    ):

        # Calculate total samples and number of batches
        original_samples_per_seed = static_syn.samples_per_seed
        total_samples = static_syn.samples_per_seed * fg_seeds_ss.n_samples * bg_seeds_ss.n_samples
        seed_num_samples = fg_seeds_ss.n_samples * bg_seeds_ss.n_samples
        num_batches = math.ceil(total_samples/batch_size)
        if verbose:
            print(f"number_of_batches: {num_batches}")

        # Keep track of how many samples you need
        total_samples_left_to_run = total_samples

        # Loop through batches
        for batch_num in range(num_batches):
            if verbose:
                print(f"\nBatch #: {batch_num}")

            # Calculate how many samples you need for this batch
            if total_samples_left_to_run >= batch_size:
                static_syn.samples_per_seed = math.ceil(batch_size / seed_num_samples)
                number_of_samples_to_get = batch_size
                total_samples_left_to_run -= batch_size
            else:
                static_syn.samples_per_seed = math.ceil(total_samples_left_to_run/seed_num_samples)
                number_of_samples_to_get = total_samples_left_to_run
                total_samples_left_to_run -= total_samples_left_to_run

            _, _, gross = static_syn.generate(
                fg_seeds_ss=fg_seeds_ss, bg_seeds_ss=bg_seeds_ss, verbose=verbose)

            if verbose:
                print(f"number_of_samples_to_get: {number_of_samples_to_get}")

            gross.normalize()
            gross_sources_cont_df = gross.sources.groupby(axis=1, level=target_level).sum()
            source_contributions_df = gross.get_source_contributions(
                target_level=target_level)
            self.model_outputs = source_contributions_df.columns.values

            for i in range(number_of_samples_to_get):
                x = copy.deepcopy(gross.spectra.values[i])
                y = gross_sources_cont_df.values[i].astype(float)
                yield x, y
        static_syn.samples_per_seed = original_samples_per_seed

    def predict(self, ss: SampleSet, bg_ss: SampleSet = None):
        """Classifies the spectra in the provided SampleSet(s).

        Results are stored inside the first SampleSet's prediction-related properties.

        Args:
            ss: Defines a SampleSet of `n` spectra where `n` >= 1 and the spectra are either
                foreground (AKA, "net") or gross.
            bg_ss: Defines a SampleSet of `n` spectra where `n` >= 1 and the spectra are background.

        """
        x_test = ss.get_samples().astype(float)
        if bg_ss:
            X = [x_test, bg_ss.get_samples().astype(float)]
        else:
            X = x_test
        results = self.model.predict(X)  # output size will be n_samples by n_labels

        col_level_idx = SampleSet.SOURCES_MULTI_INDEX_NAMES.index(self.target_level)
        col_level_subset = SampleSet.SOURCES_MULTI_INDEX_NAMES[:col_level_idx+1]
        ss.prediction_probas = pd.DataFrame(
            data=results,
            columns=pd.MultiIndex.from_tuples(
               self.model_outputs, names=col_level_subset
            )
        )

        ss.classified_by = self.info["model_id"]


class LabelProportionEstimator(TFModelBase):
    METRICS = {
        "mae": tf.metrics.MeanAbsoluteError,
        # TODO: add multi-F1 here
    }
    UNSUPERVISED_LOSS_FUNCS = {
        "poisson_nll": poisson_nll_diff,
        "normal_nll": normal_nll_diff,
        "sse": sse_diff,
        "weighted_sse": weighted_sse_diff
    }
    SUPERVISED_LOSS_FUNCS = {
        "sparsemax": (
            SparsemaxLoss,  # tfa.losses.SparsemaxLoss,
            {
                "from_logits": True,
                "reduction": tf.keras.losses.Reduction.NONE,
            },
            sparsemax,
        ),
        "categorical_crossentropy": (
            tf.keras.losses.CategoricalCrossentropy,
            {
                "from_logits": True,
                "reduction": tf.keras.losses.Reduction.NONE,
            },
            tf.keras.activations.softmax,
        ),
    }
    INFO_KEYS = (
        # model metadata
        "model_id",
        "model_type",
        "normalization",
        "pyriid_version",
        "target_level",
        # model architecture
        "hidden_layers",
        "optimizer_name",
        "learning_rate",
        "sup_loss",
        "unsup_loss",
        "beta",
        "metrics_names",
        "hidden_layer_activation",
        "l2_alpha",
        "activity_regularizer",
        "dropout",
        # dictionaries
        "fg_dict",
        # train/val histories
        "train_history",
        "val_history"
    )

    def __init__(self,
                 hidden_layers: tuple = (256,),
                 sup_loss="sparsemax",
                 unsup_loss="sse",
                 beta=0.9,
                 fg_dict=None,
                 optimizer: str = "adam",
                 learning_rate: float = 1e-3,
                 metrics: tuple = ("mae",),
                 hidden_layer_activation: str = "relu",
                 l2_alpha: float = 1e-4,
                 activity_regularizer=None,
                 dropout: float = 0.0,
                 target_level: str = "Seed",
                 train_history=None,
                 val_history=None,
                 **base_kwargs):
        """Initializes the classifier.

        The model is implemented as a tf.keras.Sequential object.

        Args:
            hidden_layers: a tuple defining the number and size of dense layers.
            sup_loss: the supervised loss function to use for training.
            unsup_loss: Define the unsupervised loss function to use for training the
                foreground branch of the network.  Options: "sse", "poisson_nll",
                "normal_nll", or "weighted_sse".
            beta: the tradeoff parameter between the supervised and unsupervised
                foreground loss.
            fg_dict: a 2D array of pure, long-collect foreground and background spectra.
            optimizer: the tensorflow optimizer or optimizer name to use for training.
            learning_rate: the learning rate for the foreground optimizer.
            metrics: a tuple of metrics to be evaluated on the foreground branch
                during training.  If a string, must be in ["categorical_crossentropy", "multi_f1"].
                Otherwise should be a standard tf.keras.metrics metric.
            hidden_layer_activation: the activate function to use for each dense layer.
            l2_alpha: the alpha value for the L2 regularization of each dense layer.
            activity_regularizer: the regularizer function applied each dense layer output.
            dropout: the amount of dropout to apply to each dense layer.
            train_history: dictionary of training history, automatically filled when loading model
            val_history: dicitionary of val history, automatically filled when loading model
        """
        super().__init__(**base_kwargs)

        self.hidden_layers = hidden_layers
        self.sup_loss = sup_loss
        self.unsup_loss = unsup_loss
        self.sup_loss_func, self.activation = self._get_sup_loss_func(
            sup_loss,
            prefix="sup"
        )
        self.sup_loss_func_name = self.sup_loss_func.name
        self.optimizer_name = optimizer
        if self.optimizer_name == "adam":
            self.optimizer = Adam(learning_rate=learning_rate)
        else:
            self.optimizer = optimizer
        self.unsup_loss_func = self._get_unsup_loss_func(unsup_loss)
        self.unsup_loss_func_name = f"unsup_{unsup_loss}_loss"
        self.beta = beta
        self.fg_dict = fg_dict
        self.semisup_loss_func_name = "semisup_loss"
        self.metrics_names = metrics
        self.metrics = self._get_initialized_metrics(metrics)
        self.model = None

        # Other properties
        self.hidden_layer_activation = hidden_layer_activation
        self.l2_alpha = l2_alpha
        self.activity_regularizer = activity_regularizer
        self.dropout = dropout
        self.target_level = target_level
        self.train_history = train_history
        self.val_history = val_history

    def _get_sup_loss_func(self, loss_func_str, prefix):
        if loss_func_str not in self.SUPERVISED_LOSS_FUNCS:
            raise KeyError(f"'{loss_func_str}' is not a supported supervised loss function.")
        func, kwargs, activation = self.SUPERVISED_LOSS_FUNCS[loss_func_str]
        loss_func_name = f"{prefix}_{loss_func_str}_loss"
        return func(name=loss_func_name, **kwargs), activation

    def _get_unsup_loss_func(self, loss_func_str):
        if loss_func_str not in self.UNSUPERVISED_LOSS_FUNCS:
            raise KeyError(f"'{loss_func_str}' is not a supported unsupervised loss function.")
        return self.UNSUPERVISED_LOSS_FUNCS[loss_func_str]

    def _get_initialized_metrics(self, metrics):
        initialized_metrics = []
        for metric in metrics:
            if metric in self.METRICS:
                initialized_metrics.append(self.METRICS[metric](name=f"{metric}"))
            else:
                initialized_metrics.append(metric)
        return initialized_metrics

    def _initialize_model(self, input_size, output_size):
        spectra_input = tf.keras.layers.Input(input_size, name="input_spectrum")
        spectra_norm = tf.keras.layers.Lambda(_l1_norm, name="normalized_input_spectrum")(
            spectra_input
        )
        x = spectra_norm
        for layer, nodes in enumerate(self.hidden_layers):
            x = tf.keras.layers.Dense(
                nodes,
                activation=self.hidden_layer_activation,
                activity_regularizer=self.activity_regularizer,
                kernel_regularizer=None,
                name=f"dense_{layer}"
            )(x)

            if self.dropout > 0:
                x = tf.keras.layers.Dropout(self.dropout)(x)
        output = tf.keras.layers.Dense(
            output_size,
            activation="linear",
            name="output"
        )(x)

        self.model = tf.keras.models.Model(
            inputs=[spectra_input],
            outputs=[output]
        )

    def _initialize_history(self):
        history = {}

        history.update({x.name: [] for x in self.metrics})
        history[self.sup_loss_func_name] = []
        history[self.unsup_loss_func_name] = []
        history[self.semisup_loss_func_name] = []

        self.train_history = history
        self.val_history = copy.deepcopy(history)

    def _initialize_trackers(self):
        self.train_trackers = {k: RunningAverage() for k in self.train_history.keys()}
        self.val_trackers = {k: RunningAverage() for k in self.val_history.keys()}

    def _update_train_history(self):
        for k, tracker in self.train_trackers.items():
            if tracker.is_empty:
                continue
            self.train_history[k].append(tracker.average)
        for m in self.metrics:
            self.train_history[m.name].append(m.result().numpy())

    def _update_val_history(self):
        for k, tracker in self.val_trackers.items():
            if tracker.is_empty:
                continue
            self.val_history[k].append(tracker.average)
        for m in self.metrics:
            self.val_history[m.name].append(m.result().numpy())

    def _get_epoch_output(self):
        epoch_output = ", ".join(
            [f"{each}: {self.train_history[each][-1]:.3f}"
                for each in self.train_history.keys()]
        )
        epoch_output += ", "
        epoch_output += ", ".join(
            [f"val_{each}: {self.val_history[each][-1]:.3f}"
                for each in self.val_history.keys()]
        )
        return epoch_output

    def _update_trackers(self, trackers, summarized_batch_results):
        for k, v in summarized_batch_results.items():
            trackers[k].add_sample(v)

    def _reset_metrics(self):
        for m in self.metrics:
            m.reset_states()

    def _get_model_file_paths(self, save_path):
        root, ext = os.path.splitext(save_path)

        if ext[1:].lower() != 'onnx':
            raise NameError("Model must be an .onnx file.")

        model_path = root + '.onnx'
        model_info_path = root + '_info.json'

        return model_info_path, model_path

    def _call_model(self, model, spectra, activation, semisup_loss, sources, training=False):
        logits = model(spectra, training=training)
        lpes = activation(logits)
        sup_losses, unsup_losses, semisup_losses = semisup_loss(spectra, sources, logits, lpes)
        return lpes, sup_losses, unsup_losses, semisup_losses

    @tf.function
    def _forward_pass(self, spectra, sources, training=False):
        with tf.GradientTape() as tape:
            lpes, sup_losses, unsup_losses, semisup_losses = \
                self._call_model(
                    self.model,
                    spectra,
                    self.activation,
                    self.semisup_loss_func,
                    sources,
                    training=training,
                )
        if training:
            # TODO: investigate whether to pass gross_losses or gross_losses.mean()
            #   to tape.gradient()
            grads = tape.gradient(
                semisup_losses,  # tf.math.reduce_sum(semisup_losses),
                self.model.trainable_weights
            )
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        for m in self.metrics:
            m.update_state(sources, lpes)

        batch_results = {
            self.sup_loss_func_name: sup_losses,
            self.unsup_loss_func_name: unsup_losses,
            self.semisup_loss_func_name: semisup_losses,
        }

        return batch_results

    def fit(self, seeds_ss: SampleSet,
            ss: SampleSet,
            batch_size: int = 10, epochs: int = 20,
            validation_split: float = 0.2, callbacks=None, patience: int = 15,
            es_monitor: str = "val_loss", es_mode: str = "min", es_verbose=0,
            target_level="Seed", verbose: bool = False):
        """Fits a model to the given SampleSet(s).

        Args:
            seeds_ss: a sampleset of pure, long-collect spectra.
            ss: a SampleSet of `n` spectra where `n` >= 1 and the spectra are gross.
            batch_size: the number of samples per gradient update.
            epochs: maximum number of training iterations.
            callbacks: a list of callbacks to be passed to TensorFlow Model.fit() method.
            patience: the number of epochs to wait for tf.keras.callbacks.EarlyStopping object.
            es_monitor: the quantity to be monitored for tf.keras.callbacks.EarlyStopping object.
            es_mode: mode for tf.keras.callbacks.EarlyStopping object.
            es_verbose: the verbosity level for tf.keras.callbacks.EarlyStopping object.
            target_level: The source level to target for model output.
            verbose: whether or not model training output is printed to the terminal.

        Returns:
            None

        Raises:
            ValueError: Raised when no spectra are provided as `ss` input.
        """
        # TODO: throw error if gross_ss and bg_ss don't have live time info - it is NEEDED for SNR

        # Gather data
        spectra = ss\
            .get_samples()\
            .astype(float)
        sources_df = ss\
            .get_source_contributions(target_level=target_level)
        sources = sources_df\
            .values.astype(float)
        self.sources_columns = sources_df.columns

        # TODO: warn if all data sums close to one (i.e., not in counts)

        # Store dictionary
        if verbose:
            print("Building dictionary...")

        if self.fg_dict is None:
            self.fg_dict = _get_reordered_spectra(
                seeds_ss.spectra,
                seeds_ss.sources,
                self.sources_columns,
                target_level=target_level
            ).values

        # Make sure the model is initialized
        if not self.model:
            if verbose:
                print("Initializing model...")
            self._initialize_model(
                ss.n_channels,
                sources.shape[1],
            )
        elif verbose:
            print("Model already initialized.")

        # Build loss function
        if verbose:
            print("Building loss functions...")

        self.semisup_loss_func = build_semisupervised_loss_func(
            self.sup_loss_func,
            self.unsup_loss_func,
            self.fg_dict,
            self.beta,
        )

        # Clear history
        self._initialize_history()

        # Fit
        if verbose:
            print("Building TF Datasets...")
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.convert_to_tensor(spectra, dtype=tf.float32),
            tf.convert_to_tensor(sources, dtype=tf.float32),
        ))

        if verbose:
            def _get_train_val_split_str(val_ratio):
                val_pct = int(val_ratio * 100)
                train_pct = 100 - val_pct
                return f"{train_pct}/{val_pct}"
            train_val_split_str = _get_train_val_split_str(validation_split)
            print(f"Splitting data {train_val_split_str} into train/val...")

        # WARNING: if you didn't shuffle your data before, here be dragons
        n_samples = ss.n_samples
        n_val_samples = int(validation_split * n_samples)
        train_dataset = dataset.skip(n_val_samples)
        val_dataset = dataset.take(n_val_samples)
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        n_train_batches = int(((1.0 - validation_split) * n_samples) / batch_size)

        if verbose:
            print("Fitting...")

        for epoch in range(epochs):
            # Training
            train_batches = tqdm(
                enumerate(train_dataset),
                unit="batch",
                postfix="loss={loss_value}",
                total=n_train_batches,
                colour='green',
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                desc=f"epoch {epoch}"
            )
            train_batches.set_postfix(loss_value=0.0)
            self._initialize_trackers()
            for _, batch in train_batches:
                batch_results = self._forward_pass(*batch, training=True)
                summarized_batch_results = {k: np.mean(v) for k, v in batch_results.items()}
                self._update_trackers(self.train_trackers, summarized_batch_results)
                train_batches.set_postfix(
                    loss_value=summarized_batch_results[self.semisup_loss_func_name]
                )
            self._update_train_history()
            self._reset_metrics()

            # Validation
            for batch in val_dataset:
                batch_results = self._forward_pass(*batch)
                summarized_batch_results = {k: np.mean(v) for k, v in batch_results.items()}
                self._update_trackers(self.val_trackers, summarized_batch_results)
            self._update_val_history()
            self._reset_metrics()

            if verbose:
                print(self._get_epoch_output())

    def predict(self, ss: SampleSet):
        """Classifies the spectra in the provided SampleSet(s).

        Results are stored inside the first SampleSet's prediction-related properties.

        Args:
            ss: a SampleSet of `n` spectra where `n` >= 1 and the spectra are gross or fg.

        """
        if ss.n_samples <= 0:
            raise ValueError("No gross/fg spectr[a|um] provided!")

        spectra = tf.convert_to_tensor(
            ss.spectra.values,
            dtype=tf.float32
        )

        self.sources_columns = ss\
            .get_source_contributions(target_level=self.target_level)\
            .columns

        # if no fitted model, use loaded onnx model
        if self.model is None:
            outputs = self.onnx_session.run(
                [self.onnx_session.get_outputs()[0].name],
                {self.onnx_session.get_inputs()[0].name: spectra.numpy()}
            )[0]
            lpes = self.activation(tf.convert_to_tensor(outputs, dtype=tf.float32))

        else:
            logits = self.model(spectra, training=False)
            lpes = self.activation(logits)

        ss.prediction_probas = pd.DataFrame(
            data=lpes,
            columns=self.sources_columns
        )

        # Fill in unsupervised losses
        recon_errors = reconstruction_error(
            spectra,
            lpes,
            self.fg_dict,
            self.unsup_loss_func
        )
        ss.info[self.unsup_loss_func_name] = recon_errors.numpy()

    def save(self, save_path) -> Tuple[str, str]:
        model_info_path, model_path = \
            self._get_model_file_paths(save_path)

        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        model_info = self.get_info_as_dict()
        model_info_df = pd.DataFrame(
            [[v] for v in model_info.values()],
            model_info.keys()
        )
        model_info_df[0].to_json(model_info_path, indent=4)

        tf2onnx.convert.from_keras(
            self.model,
            input_signature=None,
            output_path=model_path
        )

        return model_info_path, model_path

    def load(self, load_path):
        model_info_path, model_path = \
            self._get_model_file_paths(load_path)

        with open(model_info_path) as fin:
            model_info = json.load(fin)
        self.__init__(**model_info)

        self.onnx_session = onnxruntime.InferenceSession(model_path)

    def get_info_as_dict(self):
        info_dict = {k: v for k, v in vars(self).items() if k in self.INFO_KEYS}
        return info_dict
