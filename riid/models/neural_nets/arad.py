# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains implementations of the ARAD deep learning architecture."""
from typing import List

import keras
import pandas as pd
import tensorflow as tf
from keras.api.activations import sigmoid, softplus
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.initializers import GlorotNormal, HeNormal
from keras.api.layers import (BatchNormalization, Concatenate, Conv1D,
                              Conv1DTranspose, Dense, Dropout, Flatten, Input,
                              MaxPool1D, Reshape, UpSampling1D)
from keras.api.losses import kl_divergence, log_cosh
from keras.api.metrics import MeanSquaredError
from keras.api.models import Model
from keras.api.optimizers import Adam, Nadam
from keras.api.regularizers import L1L2, L2
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from riid import SampleSet, SpectraState
from riid.losses import mish
from riid.models.base import PyRIIDModel
from riid.models.layers import ExpandDimsLayer


class ARADv1TF(Model):
    """TensorFlow Implementation of ARAD v1.

    This implementation was built based on our interpretation of the methods outlined in the
    following paper and may not fully represent the original implementation:

    - Ghawaly Jr, James M. "A Datacentric Algorithm for Gamma-ray Radiation Anomaly Detection
      in Unknown Background Environments." (2020).
    """
    def __init__(self, latent_dim: int = 5, **kwargs):
        """
        Args:
            latent_dim: dimension of internal latent represention.
                5 was the final one in the paper, but 4 to 8 were found to work well.
        """
        super().__init__(**kwargs)

        input_size = (128,)
        # Encoder
        b1_config = (
            (5, 1, 32),
            (6, 2, 16),
            (6, 2, 8),
            (4, 2, 4),
            (1, 1, 2),
        )
        b2_config = (
            (10, 2, 32),
            (6, 2, 16),
            (4, 2, 8),
            (1, 1, 4),
            (1, 1, 2),
        )
        b3_config = (
            (3, 1, 32),
            (6, 2, 16),
            (3, 2, 8),
            (4, 2, 4),
            (1, 1, 2),
        )
        encoder_input = Input(shape=input_size, name="encoder_input")
        expanded_encoder_input = ExpandDimsLayer()(encoder_input, axis=-1)
        b1 = self._get_branch(expanded_encoder_input, b1_config, 0.1, "softplus", "B1", 5)
        b2 = self._get_branch(expanded_encoder_input, b2_config, 0.1, "softplus", "B2", 5)
        b3 = self._get_branch(expanded_encoder_input, b3_config, 0.1, "softplus", "B3", 5)
        x = Concatenate(axis=1)([b1, b2, b3])
        x = Reshape((15,), name="reshape")(x)
        latent_space = Dense(
            units=latent_dim,
            name="D1_latent_space",
            activation=softplus
        )(x)
        encoder_output = BatchNormalization(name="D1_batch_norm")(latent_space)
        encoder = Model(encoder_input, encoder_output, name="encoder")

        # Decoder
        decoder_input = Input(shape=(latent_dim,), name="decoder_input")
        x = Dense(units=40, name="D2", activation=softplus)(decoder_input)
        x = Dropout(rate=0.1, name="D2_dropout")(x)
        decoder_output = Dense(
            units=128,
            activation=softplus,
            name="D3"
        )(x)
        decoder = Model(decoder_input, decoder_output, name="decoder")

        # Autoencoder
        encoded_spectrum = encoder(encoder_input)
        decoded_spectrum = decoder(encoded_spectrum)
        autoencoder = Model(encoder_input, decoded_spectrum, name="autoencoder")

        self.encoder = encoder
        self.decoder = decoder
        self.autoencoder = autoencoder
        self.sparsities = [0.001] * latent_dim
        self.penalty_weight = 0.5

    def _get_branch(self, input_layer, config, dropout_rate, activation, branch_name, dense_units):
        x = input_layer
        for i, (kernel_size, strides, filters) in enumerate(config, start=1):
            layer_name = f"{branch_name}_C{i}"
            x = Conv1D(kernel_size=kernel_size, strides=strides, filters=filters,
                       activation=activation, name=layer_name)(x)
            x = BatchNormalization(name=f"{layer_name}_batch_norm")(x)
            x = Dropout(rate=dropout_rate, name=f"{layer_name}_dropout")(x)
        x = Flatten(name=f"{branch_name}_flatten")(x)
        x = Dense(
            units=dense_units,
            name=f"{branch_name}_D1",
            activation=activation
        )(x)
        x = BatchNormalization(name=f"{branch_name}_batch_norm")(x)
        return x

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Compute loss
        logcosh_loss = log_cosh(x, decoded)
        kld_loss = kl_divergence(self.sparsities, encoded)
        loss = logcosh_loss + self.penalty_weight * kld_loss
        self.add_loss(loss)

        return decoded


class ARADv2TF(Model):
    """TensorFlow Implementation of ARAD v2.

    This implementation of ARAD was built based on our interpretation of the methods outlined in the
    following paper and may not fully represent the original implementation:

    - Ghawaly Jr, James M., et al. "Characterization of the Autoencoder Radiation Anomaly Detection
      (ARAD) model." Engineering Applications of Artificial Intelligence 111 (2022): 104761.
    """
    def __init__(self, latent_dim: int = 8, **kwargs):
        """
        Args:
            latent_dim: dimension of internal latent represention.
                5 was the final one in the paper, but 4 to 8 were found to work well.
        """
        super().__init__(**kwargs)

        input_size = (128,)
        # Encoder
        config = (
            (7, 1, 8, 2),
            (5, 1, 8, 2),
            (3, 1, 8, 2),
            (3, 1, 8, 2),
            (3, 1, 8, 2),
        )
        encoder_input = Input(shape=input_size, name="encoder_input")
        expanded_encoder_input = ExpandDimsLayer()(encoder_input, axis=-1)
        x = expanded_encoder_input
        for i, (kernel_size, strides, filters, max_pool_size) in enumerate(config, start=1):
            conv_name = f"conv{i}"
            x = Conv1D(
                kernel_size=kernel_size,
                strides=strides,
                filters=filters,
                padding="same",
                activation=mish,
                kernel_regularizer=L1L2(l1=1e-3, l2=1e-3),
                bias_regularizer=L2(l2=1e-3),
                kernel_initializer=HeNormal,
                name=conv_name)(x)
            x = BatchNormalization(name=f"{conv_name}_batch_norm")(x)
            pool_name = f"MP{i}"
            x = MaxPool1D(pool_size=max_pool_size, name=pool_name)(x)
            x = BatchNormalization(name=f"{pool_name}_batch_norm")(x)

        x = Flatten(name="flatten")(x)
        x = Dense(
            units=latent_dim,
            activation=mish,
            kernel_regularizer=L1L2(l1=1e-3, l2=1e-3),
            bias_regularizer=L2(l2=1e-3),
            kernel_initializer=HeNormal,
            name="D1"
        )(x)
        encoder_output = BatchNormalization(name="D1_batch_norm")(x)
        encoder = Model(encoder_input, encoder_output, name="encoder")

        # Decoder
        decoder_input = Input(shape=(latent_dim,), name="decoder_input")
        x = Dense(units=32, name="D2", activation=mish)(decoder_input)
        x = BatchNormalization(name="D2_batch_norm")(x)
        x = Reshape((4, 8), name="reshape")(x)
        reversed_config = enumerate(reversed(config[1:]), start=1)
        for i, (kernel_size, strides, filters, max_pool_size) in reversed_config:
            upsample_name = f"US{i}"
            x = UpSampling1D(size=max_pool_size, name=upsample_name)(x)
            x = BatchNormalization(name=f"{upsample_name}_batch_norm")(x)
            conv_name = f"tconv{i}"
            x = Conv1D(
                kernel_size=kernel_size,
                strides=strides,
                filters=filters,
                padding="same",
                activation=mish,
                kernel_regularizer=L1L2(l1=1e-3, l2=1e-3),
                bias_regularizer=L2(l2=1e-3),
                kernel_initializer=HeNormal,
                name=conv_name)(x)
            x = BatchNormalization(name=f"{conv_name}_batch_norm")(x)

        i += 1
        upsample_name = f"US{i}"
        x = UpSampling1D(size=max_pool_size, name=upsample_name)(x)
        x = BatchNormalization(name=f"{upsample_name}_batch_norm")(x)
        x = Conv1DTranspose(
            kernel_size=7,
            strides=1,
            filters=1,
            padding="same",
            activation=sigmoid,
            kernel_initializer=GlorotNormal,
            name=f"tconv{i}"
        )(x)
        decoder_output = Reshape((128,), name="reshape_final")(x)
        decoder = Model(decoder_input, decoder_output, name="decoder")

        # Autoencoder
        encoded_spectrum = encoder(encoder_input)
        decoded_spectrum = decoder(encoded_spectrum)
        autoencoder = Model(encoder_input, decoded_spectrum, name="autoencoder")

        self.encoder = encoder
        self.decoder = decoder
        self.autoencoder = autoencoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Compute loss
        p_sum = tf.reduce_sum(x, axis=-1)
        p_norm = tf.divide(
            x,
            tf.reshape(p_sum, (-1, 1))
        )
        q_sum = tf.reduce_sum(decoded, axis=-1)
        q_norm = tf.divide(
            decoded,
            tf.reshape(q_sum, (-1, 1))
        )
        m = (p_norm + q_norm) / 2
        js_divergence = (kl_divergence(p_norm, m) + kl_divergence(q_norm, m)) / 2
        loss = tf.math.sqrt(js_divergence)
        self.add_loss(loss)

        return decoded


class ARADv1(PyRIIDModel):
    """PyRIID-compatible ARAD v1 model supporting `SampleSet`s.
    """
    def __init__(self, model: ARADv1TF = None):
        """
        Args:
            model: a previously initialized TF implementation of ARADv1
        """
        super().__init__()

        self.model = model

        self._update_custom_objects("ARADv1TF", ARADv1TF)

    def fit(self, ss: SampleSet, epochs: int = 300, validation_split=0.2,
            es_verbose: int = 0, verbose: bool = False):
        """Fit a model to the given `SampleSet`.

        Args:
            ss: `SampleSet` of `n` spectra where `n` >= 1
            epochs: maximum number of training epochs
            validation_split: percentage of the training data to use as validation data
            es_verbose: verbosity level for `tf.keras.callbacks.EarlyStopping`
            verbose: whether to show detailed model training output

        Returns:
            reconstructed_spectra: output of ARAD model
        """
        _check_spectra(ss)

        x = ss.get_samples().astype(float)

        optimizer = Nadam(learning_rate=1e-4)

        if not self.model:
            self.model = ARADv1TF()

        self.model.compile(optimizer=optimizer)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                verbose=es_verbose,
                restore_best_weights=True,
                mode="min",
                min_delta=1e-7
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=3,
                min_delta=1e-8
            )
        ]

        history = self.model.fit(
            x=x,
            y=None,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split,
            callbacks=callbacks,
            shuffle=True,
            batch_size=64
        )

        self._update_info(
            normalization=ss.spectra_state,
        )

        return history

    def predict(self, ss: SampleSet, verbose=False):
        """Generate reconstructions for given `SampleSet`.

        Args:
            ss: `SampleSet` of `n` spectra where `n` >= 1

        Returns:
            reconstructed_spectra: output of ARAD model
        """
        _check_spectra(ss)

        x = ss.get_samples().astype(float)

        reconstructed_spectra = self.model.predict(x, verbose=verbose)
        reconstruction_errors = entropy(x, reconstructed_spectra, axis=1)
        ss.info["recon_error"] = reconstruction_errors

        return reconstructed_spectra


class ARADv2(PyRIIDModel):
    """PyRIID-compatible ARAD v2 model supporting `SampleSet`s.
    """
    def __init__(self, model: ARADv2TF = None):
        """
        Args:
            model: a previously initialized TF implementation of ARADv1
        """
        super().__init__()

        self.model = model

        self._update_custom_objects("ARADv2TF", ARADv2TF)

    def fit(self, ss: SampleSet, epochs: int = 300, validation_split=0.2,
            es_verbose: int = 0, verbose: bool = False):
        """Fit a model to the given `SampleSet`.

        Args:
            ss: `SampleSet` of `n` spectra where `n` >= 1
            epochs: maximum number of training epochs
            validation_split: percentage of the training data to use as validation data
            es_verbose: verbosity level for `tf.keras.callbacks.EarlyStopping`
            verbose: whether to show detailed model training output

        Returns:
            reconstructed_spectra: output of ARAD model
        """
        _check_spectra(ss)

        x = ss.get_samples().astype(float)

        optimizer = Adam(learning_rate=0.01, epsilon=0.05)

        if not self.model:
            self.model = ARADv2TF()

        self.model.compile(optimizer=optimizer)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=6,
                verbose=es_verbose,
                restore_best_weights=True,
                mode="min",
                min_delta=1e-4
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=3,
                min_delta=1e-4
            )
        ]

        history = self.model.fit(
            x=x,
            y=x,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split,
            callbacks=callbacks,
            shuffle=True,
            batch_size=32
        )

        self._update_info(
            normalization=ss.spectra_state,
        )

        return history

    def predict(self, ss: SampleSet, verbose=False):
        """Generate reconstructions for given `SampleSet`.

        Args:
            ss: `SampleSet` of `n` spectra where `n` >= 1

        Returns:
            reconstructed_spectra: output of ARAD model
        """
        _check_spectra(ss)

        x = ss.get_samples().astype(float)

        reconstructed_spectra = self.model.predict(x, verbose=verbose)
        reconstruction_errors = jensenshannon(x, reconstructed_spectra, axis=1)
        ss.info["recon_error"] = reconstruction_errors

        return reconstructed_spectra


class ARADLatentPredictor(PyRIIDModel):
    """PyRIID-compatible model for branching from the latent space of a pre-trained
    ARAD model for a separate, arbitrary prediction task.
    """
    def __init__(self, hidden_layers: tuple = (8, 4,),
                 hidden_activation: str = "relu", final_activation: str = "linear",
                 loss: str = "mse", optimizer="adam", optimizer_kwargs=None,
                 learning_rate: float = 1e-3, metrics=None,
                 kernel_l1_regularization: float = 0.0, kernel_l2_regularization: float = 0.0,
                 bias_l1_regularization: float = 0.0, bias_l2_regularization: float = 0.0,
                 activity_l1_regularization: float = 0.0, activity_l2_regularization: float = 0.0,
                 dropout: float = 0.0, **base_kwargs):
        """
        Args:
            hidden_layers: tuple defining the number and size of dense layers
            hidden_activation: activation function to use for each dense layer
            final_activation: activation function to use for final layer
            loss: loss function to use for training
            optimizer: tensorflow optimizer or optimizer name to use for training
            optimizer_kwargs: kwargs for optimizer
            learning_rate: optional learning rate for the optimizer
            metrics: list of metrics to be evaluating during training
            kernel_l1_regularization: l1 regularization value for the kernel regularizer
            kernel_l2_regularization: l2 regularization value for the kernel regularizer
            bias_l1_regularization: l1 regularization value for the bias regularizer
            bias_l2_regularization: l2 regularization value for the bias regularizer
            activity_l1_regularization: l1 regularization value for the activity regularizer
            activity_l2_regularization: l2 regularization value for the activity regularizer
            dropout: amount of dropout to apply to each dense layer
        """
        super().__init__(**base_kwargs)

        self.hidden_layers = hidden_layers
        self.hidden_activation = hidden_activation
        self.final_activation = final_activation
        self.loss = loss
        self.optimizer = optimizer
        if isinstance(optimizer, str):
            self.optimizer = keras.optimizers.get(optimizer)
        if optimizer_kwargs is not None:
            for key, value in optimizer_kwargs.items():
                setattr(self.optimizer, key, value)
        self.optimizer.learning_rate = learning_rate
        self.metrics = metrics
        if self.metrics is None:
            self.metrics = [MeanSquaredError()]
        self.kernel_l1_regularization = kernel_l1_regularization
        self.kernel_l2_regularization = kernel_l2_regularization
        self.bias_l1_regularization = bias_l1_regularization
        self.bias_l2_regularization = bias_l2_regularization
        self.activity_l1_regularization = activity_l1_regularization
        self.activity_l2_regularization = activity_l2_regularization
        self.dropout = dropout
        self.model = None
        self.encoder = None

    def _initialize_model(self, arad: Model, output_size: int):
        """Build Keras MLP model.
        """
        encoder: Model = arad.get_layer("encoder")
        encoder_input = encoder.input
        encoder_output = encoder.output
        encoder_output_shape = encoder_output.shape[-1]

        predictor_input = Input(shape=(encoder_output_shape,), name="inner_predictor_input")
        x = predictor_input
        for layer, nodes in enumerate(self.hidden_layers):
            x = Dense(
                nodes,
                activation=self.hidden_activation,
                kernel_regularizer=L1L2(
                    l1=self.kernel_l1_regularization,
                    l2=self.kernel_l2_regularization
                ),
                bias_regularizer=L1L2(
                    l1=self.bias_l1_regularization,
                    l2=self.bias_l2_regularization
                ),
                activity_regularizer=L1L2(
                    l1=self.activity_l1_regularization,
                    l2=self.activity_l2_regularization
                ),
                name=f"inner_predictor_dense_{layer}"
            )(x)
            if self.dropout > 0:
                x = Dropout(self.dropout)(x)
        predictor_output = Dense(
            output_size,
            activation=self.final_activation,
            name="inner_predictor_output"
        )(x)
        inner_predictor = Model(predictor_input, predictor_output, name="inner_predictor")

        encoded_spectrum = encoder(encoder_input)
        predictions = inner_predictor(encoded_spectrum)
        self.model = Model(encoder_input, predictions, name="predictor")
        # Freeze the layers corresponding to the autoencoder
        # Note: setting trainable to False is recursive to sub-layers per TF docs:
        # https://www.tensorflow.org/guide/keras/transfer_learning#recursive_setting_of_the_trainable_attribute
        for layer in self.model.layers[:-1]:
            layer.trainable = False

    def _check_targets(self, target_info_columns, target_level):
        """Check that valid target options are provided."""
        if target_info_columns and target_level:
            raise ValueError((
                "You have specified both target_info_columns (regression task) and "
                "a target_level (classification task), but only one can be set."
            ))
        if not target_info_columns and not target_level:
            raise ValueError((
                "You must specify either target_info_columns (regression task) or "
                "a target_level (classification task)."
            ))

    def fit(self, arad: Model, ss: SampleSet, target_info_columns: List[str] = None,
            target_level: str = None, batch_size: int = 10, epochs: int = 20,
            validation_split: float = 0.2, callbacks=None, patience: int = 15,
            es_monitor: str = "val_loss", es_mode: str = "min", es_verbose=0,
            es_min_delta: float = 0.0, verbose: bool = False):
        """Fit a model to the given SampleSet(s).

        Args:
            arad: a pretrained ARAD model (a TensorFlow Model object, not a PyRIIDModel wrapper)
            ss: `SampleSet` of `n` spectra where `n` >= 1
            target_info_columns: list of columns names from SampleSet info dataframe which
                denote what values the model should target
            target_level: `SampleSet.sources` column level to target for classification
            batch_size: number of samples per gradient update
            epochs: maximum number of training iterations
            validation_split: proportion of training data to use as validation data
            callbacks: list of callbacks to be passed to TensorFlow Model.fit() method
            patience: number of epochs to wait for tf.keras.callbacks.EarlyStopping object
            es_monitor: quantity to be monitored for tf.keras.callbacks.EarlyStopping object
            es_mode: mode for tf.keras.callbacks.EarlyStopping object
            es_verbose: verbosity level for tf.keras.callbacks.EarlyStopping object
            es_min_delta: minimum change to count as an improvement for early stopping
            verbose: whether model training output is printed to the terminal
        """
        self._check_targets(target_info_columns, target_level)

        x_train = ss.get_samples().astype(float)
        if target_info_columns:
            y_train = ss.info[target_info_columns].values.astype(float)
        else:
            source_contributions_df = ss.sources.groupby(
                axis=1,
                level=target_level,
                sort=False
            ).sum()
            y_train = source_contributions_df.values.astype(float)

        if not self.model:
            self._initialize_model(arad=arad, output_size=y_train.shape[1])

        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics
        )
        es = EarlyStopping(
            monitor=es_monitor,
            patience=patience,
            verbose=es_verbose,
            restore_best_weights=True,
            mode=es_mode,
            min_delta=es_min_delta
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
            callbacks=callbacks,
            shuffle=True,
            batch_size=batch_size
        )

        self._update_info(
            normalization=ss.spectra_state,
            target_level=target_level,
            model_outputs=target_info_columns,
        )
        if target_level:
            self._update_info(
                model_outputs=source_contributions_df.columns.values.tolist(),
            )

        return history

    def predict(self, ss: SampleSet, verbose=False):
        spectra = ss.get_samples().astype(float)
        predictions = self.model.predict(spectra, verbose=verbose)

        if self.target_level:
            col_level_idx = SampleSet.SOURCES_MULTI_INDEX_NAMES.index(self.target_level)
            col_level_subset = SampleSet.SOURCES_MULTI_INDEX_NAMES[:col_level_idx+1]
            ss.prediction_probas = pd.DataFrame(
                data=predictions,
                columns=pd.MultiIndex.from_tuples(
                    self.get_model_outputs_as_label_tuples(),
                    names=col_level_subset
                )
            )

        ss.classified_by = self.model_id

        return predictions


def _check_spectra(ss: SampleSet):
    """Checks if SampleSet spectra are compatible with ARAD models."""
    if ss.n_samples <= 0:
        raise ValueError("No spectr[a|um] provided!")
    if not ss.all_spectra_sum_to_one():
        raise ValueError("All spectra must sum to one.")
    if not ss.spectra_state == SpectraState.L1Normalized:
        raise ValueError(
            f"SpectraState must be L1Normalzied, provided SpectraState is {ss.spectra_state}."
        )
    if not ss.n_channels == 128:
        raise ValueError(
            f"Spectra must have 128 channels, provided spectra have {ss.n_channels} channels."
        )
