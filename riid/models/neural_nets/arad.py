# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains implementations of the ARAD model architecture."""
import tensorflow as tf
from keras.activations import sigmoid
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.initializers import GlorotNormal, HeNormal
from keras.layers import (BatchNormalization, Concatenate, Conv1D,
                          Conv1DTranspose, Dense, Dropout, Flatten, Input,
                          MaxPool1D, Reshape, UpSampling1D)
from keras.models import Model
from keras.regularizers import L1L2, L2
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from riid.data.sampleset import SampleSet, SpectraState
from riid.losses import jensen_shannon_distance, mish
from riid.models import PyRIIDModel


class ARADv1TF(Model):
    """TensorFlow Implementation of ARAD v1.

    This implementation was built based on our interpretation of the methods outlined in the
    following paper and may not fully represent the original implementation:

    - Ghawaly Jr, James M. "A Datacentric Algorithm for Gamma-ray Radiation Anomaly Detection
      in Unknown Background Environments." (2020).
    """
    def __init__(self, latent_dim: int = 5):
        """
        Args:
            latent_dim: dimension of internal latent represention.
                5 was the final one in the paper, but 4 to 8 were found to work well.
        """
        super().__init__()

        input_size = (128, 1)
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
        b1 = self._get_branch(encoder_input, b1_config, 0.1, "softplus", "B1", 5)
        b2 = self._get_branch(encoder_input, b2_config, 0.1, "softplus", "B2", 5)
        b3 = self._get_branch(encoder_input, b3_config, 0.1, "softplus", "B3", 5)
        x = Concatenate(axis=1)([b1, b2, b3])
        x = Reshape((15,), name="reshape")(x)
        latent_space = Dense(units=latent_dim, name="D1_latent_space")(x)
        encoder_output = BatchNormalization(name="D1_batch_norm")(latent_space)
        encoder = Model(encoder_input, encoder_output, name="encoder")

        # Decoder
        decoder_input = Input(shape=(latent_dim,), name="decoder_input")
        x = Dense(units=40, name="D2")(decoder_input)
        x = Dropout(rate=0.1, name="D2_dropout")(x)
        decoder_output = Dense(
            units=128,
            activation=sigmoid,  # unclear from paper, seems to be necessary
            name="D3"
        )(x)
        decoder = Model(decoder_input, decoder_output, name="decoder")

        # Autoencoder
        encoded_spectrum = encoder(encoder_input)
        decoded_spectrum = decoder(encoded_spectrum)
        autoencoder = Model(encoder_input, decoded_spectrum, name="autoencoder")

        def logcosh_with_kld_penalty(input_spectrum, decoded_spectrum,
                                     latent_space, sparsities,
                                     penalty_weight=0.5):
            squeezed_input = tf.squeeze(input_spectrum)
            logcosh_loss = tf.keras.losses.log_cosh(squeezed_input, decoded_spectrum)
            kld_loss = tf.keras.losses.kld(sparsities, latent_space)
            loss = logcosh_loss + penalty_weight * kld_loss
            return loss

        sparsities = [0.001] * latent_dim
        loss_func = logcosh_with_kld_penalty(
            encoder_input,
            decoded_spectrum,
            latent_space,
            sparsities,
        )
        autoencoder.add_loss(loss_func)

        self.encoder = encoder
        self.decoder = decoder
        self.autoencoder = autoencoder

    def _get_branch(self, input_layer, config, dropout_rate, activation, branch_name, dense_units):
        x = input_layer
        for i, (kernel_size, strides, filters) in enumerate(config, start=1):
            layer_name = f"{branch_name}_C{i}"
            x = Conv1D(kernel_size=kernel_size, strides=strides, filters=filters,
                       activation=activation, name=layer_name)(x)
            x = BatchNormalization(name=f"{layer_name}_batch_norm")(x)
            x = Dropout(rate=dropout_rate, name=f"{layer_name}_dropout")(x)
        x = Flatten(name=f"{branch_name}_flatten")(x)
        x = Dense(units=dense_units, name=f"{branch_name}_D1")(x)
        x = BatchNormalization(name=f"{branch_name}_batch_norm")(x)
        return x

    def call(self, x):
        decoded = self.autoencoder(x)
        return decoded


class ARADv2TF(Model):
    """TensorFlow Implementation of ARAD v2.

    This implementation of ARAD was built based on our interpretation of the methods outlined in the
    following paper and may not fully represent the original implementation:

    - Ghawaly Jr, James M., et al. "Characterization of the Autoencoder Radiation Anomaly Detection
      (ARAD) model." Engineering Applications of Artificial Intelligence 111 (2022): 104761.
    """
    def __init__(self, latent_dim: int = 8):
        """
        Args:
            latent_dim: dimension of internal latent represention.
                5 was the final one in the paper, but 4 to 8 were found to work well.
        """
        super().__init__()

        input_size = (128, 1)
        # Encoder
        config = (
            (7, 1, 8, 2),
            (5, 1, 8, 2),
            (3, 1, 8, 2),
            (3, 1, 8, 2),
            (3, 1, 8, 2),
        )
        encoder_input = Input(shape=input_size, name="encoder_input")
        x = encoder_input
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
        decoder_input = Input(shape=latent_dim, name="decoder_input")
        x = Dense(units=32, name="D2")(decoder_input)
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
        decoded = self.autoencoder(x)
        return decoded


class ARAD(PyRIIDModel):
    """PyRIID-compatible ARAD model to work with SampleSets.
    """
    def __init__(self, model: Model = ARADv2TF()):
        """
        Args:
            model: instantiated model of the desired version of ARAD to use.
        """
        super().__init__()

        self.model = model

    def _check_spectra(self, ss):
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
        self._check_spectra(ss)

        x = ss.get_samples().astype(float)

        is_v1 = isinstance(self.model, ARADv1TF)
        is_v2 = isinstance(self.model, ARADv2TF)
        if is_v1:
            optimizer = tf.keras.optimizers.Nadam(
                learning_rate=1e-4
            )
            loss_func = None
            es_patience = 5
            es_min_delta = 1e-7
            lr_sched_patience = 3
            lr_sched_min_delta = 1e-8
            batch_size = 64
            y = None
        elif is_v2:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.01,
                epsilon=0.05
            )
            loss_func = jensen_shannon_distance
            es_patience = 6
            es_min_delta = 1e-4
            lr_sched_patience = 3
            lr_sched_min_delta = 1e-4
            batch_size = 32
            y = x
        else:
            raise ValueError("Invalid model provided, must be ARADv1TF or ARADv2TF.")

        self.model.compile(
            loss=loss_func,
            optimizer=optimizer
        )
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=es_patience,
                verbose=es_verbose,
                restore_best_weights=True,
                mode="min",
                min_delta=es_min_delta
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=lr_sched_patience,
                min_delta=lr_sched_min_delta
            )
        ]

        history = self.model.fit(
            x=x,
            y=y,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split,
            callbacks=callbacks,
            shuffle=True,
            batch_size=batch_size
        )
        self.history = history.history

        return history

    def predict(self, ss: SampleSet, verbose=False):
        """Generate reconstructions for given `SampleSet`.

        Args:
            ss: `SampleSet` of `n` spectra where `n` >= 1

        Returns:
            reconstructed_spectra: output of ARAD model
        """
        self._check_spectra(ss)

        x = ss.get_samples().astype(float)

        reconstructed_spectra = self.get_predictions(x, verbose=verbose)

        is_v1 = isinstance(self.model, ARADv1TF)
        is_v2 = isinstance(self.model, ARADv2TF)
        if is_v1:
            # Entropy is equivalent to KL Divergence with how it is used here
            reconstruction_metric = entropy
        elif is_v2:
            reconstruction_metric = jensenshannon

        reconstruction_errors = reconstruction_metric(x, reconstructed_spectra, axis=1)
        ss.info["recon_error"] = reconstruction_errors

        return reconstructed_spectra
