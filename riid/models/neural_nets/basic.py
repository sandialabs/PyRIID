# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains a simple neural network."""
import pandas as pd
import tensorflow as tf
from keras.api.callbacks import EarlyStopping
from keras.api.layers import Dense, Input, Dropout
from keras.api.losses import CategoricalCrossentropy
from keras.api.metrics import F1Score, Precision, Recall
from keras.api.models import Model
from keras.api.optimizers import Adam
from keras.api.regularizers import l1, l2
from keras.api.utils import split_dataset

from riid import SampleSet, SpectraType
from riid.models.base import ModelInput, PyRIIDModel


class MLPClassifier(PyRIIDModel):
    """Multi-layer perceptron classifier."""
    def __init__(self, activation=None, loss=None, optimizer=None,
                 metrics=None, l2_alpha: float = 1e-4,
                 activity_regularizer=None, final_activation=None,
                 dense_layer_size=None, dropout=None):
        """
        Args:
            activation: activate function to use for each dense layer
            loss: loss function to use for training
            optimizer: tensorflow optimizer or optimizer name to use for training
            metrics: list of metrics to be evaluating during training
            l2_alpha: alpha value for the L2 regularization of each dense layer
            activity_regularizer: regularizer function applied each dense layer output
            final_activation: final activation function to apply to model output
        """
        super().__init__()

        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.final_activation = final_activation
        self.metrics = metrics
        self.l2_alpha = l2_alpha
        self.activity_regularizer = activity_regularizer
        self.final_activation = final_activation
        self.dense_layer_size = dense_layer_size
        self.dropout = dropout

        if self.activation is None:
            self.activation = "relu"
        if self.loss is None:
            self.loss = CategoricalCrossentropy()
        if optimizer is None:
            self.optimizer = Adam(learning_rate=0.01, clipnorm=0.001)
        if self.metrics is None:
            self.metrics = [F1Score(), Precision(), Recall()]
        if self.activity_regularizer is None:
            self.activity_regularizer = l1(0.0)
        if self.final_activation is None:
            self.final_activation = "softmax"
        self.model = None
        self._predict_fn = None

    def fit(self, ss: SampleSet, batch_size: int = 200, epochs: int = 20,
            validation_split: float = 0.2, callbacks=None,
            patience: int = 15, es_monitor: str = "val_loss",
            es_mode: str = "min", es_verbose=0, target_level="Isotope", verbose: bool = False):
        """Fit a model to the given `SampleSet`(s).

        Args:
            ss: `SampleSet` of `n` spectra where `n` >= 1 and the spectra are either
                foreground (AKA, "net") or gross.
            batch_size: number of samples per gradient update
            epochs: maximum number of training iterations
            validation_split: percentage of the training data to use as validation data
            callbacks: list of callbacks to be passed to the TensorFlow `Model.fit()` method
            patience: number of epochs to wait for `EarlyStopping` object
            es_monitor: quantity to be monitored for `EarlyStopping` object
            es_mode: mode for `EarlyStopping` object
            es_verbose: verbosity level for `EarlyStopping` object
            target_level: `SampleSet.sources` column level to use
            verbose: whether to show detailed model training output

        Returns:
            `tf.History` object.

        Raises:
            `ValueError` when no spectra are provided as input
        """
        if ss.n_samples <= 0:
            raise ValueError("No spectr[a|um] provided!")

        if ss.spectra_type == SpectraType.Gross:
            self.model_inputs = (ModelInput.GrossSpectrum,)
        elif ss.spectra_type == SpectraType.Foreground:
            self.model_inputs = (ModelInput.ForegroundSpectrum,)
        elif ss.spectra_type == SpectraType.Background:
            self.model_inputs = (ModelInput.BackgroundSpectrum,)
        else:
            raise ValueError(f"{ss.spectra_type} is not supported in this model.")

        X = ss.get_samples()
        source_contributions_df = ss.sources.T.groupby(target_level, sort=False).sum().T
        model_outputs = source_contributions_df.columns.values.tolist()
        Y = source_contributions_df.values

        spectra_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        labels_tensor = tf.convert_to_tensor(Y, dtype=tf.float32)
        training_dataset = tf.data.Dataset.from_tensor_slices((spectra_tensor, labels_tensor))
        training_dataset, validation_dataset = split_dataset(
            training_dataset,
            left_size=validation_split,
            shuffle=True
        )
        training_dataset = training_dataset.batch(batch_size=batch_size)
        validation_dataset = validation_dataset.batch(batch_size=batch_size)

        if not self.model:
            inputs = Input(shape=(X.shape[1],), name="Spectrum")
            if self.dense_layer_size is None:
                dense_layer_size = X.shape[1] // 2
            else:
                dense_layer_size = self.dense_layer_size
            dense_layer = Dense(
                dense_layer_size,
                activation=self.activation,
                activity_regularizer=self.activity_regularizer,
                kernel_regularizer=l2(self.l2_alpha),
            )(inputs)
            if self.dropout is not None:
                last_layer = Dropout(0.2)(dense_layer)
            else:
                last_layer = dense_layer
            outputs = Dense(Y.shape[1], activation=self.final_activation)(last_layer)
            self.model = Model(inputs, outputs)
            self.model.compile(loss=self.loss, optimizer=self.optimizer,
                               metrics=self.metrics)

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
            training_dataset,
            epochs=epochs,
            verbose=verbose,
            validation_data=validation_dataset,
            callbacks=callbacks,
         )

        # Update model information
        self._update_info(
            target_level=target_level,
            model_outputs=model_outputs,
            normalization=ss.spectra_state,
        )

        # Define the predict function with tf.function and input_signature
        self._predict_fn = tf.function(
            self._predict,
            # input_signature=[tf.TensorSpec(shape=[None, X.shape[1]], dtype=tf.float32)]
            experimental_relax_shapes=True
        )

        return history

    def _predict(self, input_tensor):
        return self.model(input_tensor, training=False)

    def predict(self, ss: SampleSet, bg_ss: SampleSet = None):
        """Classify the spectra in the provided `SampleSet`(s).

        Results are stored inside the first SampleSet's prediction-related properties.

        Args:
            ss: `SampleSet` of `n` spectra where `n` >= 1 and the spectra are either
                foreground (AKA, "net") or gross
            bg_ss: `SampleSet` of `n` spectra where `n` >= 1 and the spectra are background
        """
        x_test = ss.get_samples().astype(float)
        if bg_ss:
            X = [x_test, bg_ss.get_samples().astype(float)]
        else:
            X = x_test

        spectra_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        results = self._predict_fn(spectra_tensor)

        col_level_idx = SampleSet.SOURCES_MULTI_INDEX_NAMES.index(self.target_level)
        col_level_subset = SampleSet.SOURCES_MULTI_INDEX_NAMES[:col_level_idx+1]
        ss.prediction_probas = pd.DataFrame(
            data=results,
            columns=pd.MultiIndex.from_tuples(
                self.get_model_outputs_as_label_tuples(),
                names=col_level_subset
            )
        )

        ss.classified_by = self.model_id
