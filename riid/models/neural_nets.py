# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains a multi-layer perceptron classifier."""
from typing import Any, List

import numpy as np
import pandas as pd
import tensorflow as tf
from riid.data import SampleSet
from riid.models import ModelInput, TFModelBase
from riid.models.metrics import multi_f1, single_f1
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2


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
