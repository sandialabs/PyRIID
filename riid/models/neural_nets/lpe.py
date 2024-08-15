# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains the label proportion estimator."""

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.activations import sigmoid, softmax
from keras.api.callbacks import EarlyStopping
from keras.api.layers import Dense, Dropout, Input
from keras.api.losses import CategoricalCrossentropy, MeanSquaredError
from keras.api.models import Model
from keras.api.regularizers import L1L2
from scipy.interpolate import UnivariateSpline

from riid import SampleSet
from riid.losses import (build_keras_semisupervised_loss_func,
                         chi_squared_diff, jensen_shannon_divergence,
                         normal_nll_diff, poisson_nll_diff,
                         reconstruction_error, sse_diff, weighted_sse_diff)
from riid.losses.sparsemax import SparsemaxLoss, sparsemax
from riid.metrics import build_keras_semisupervised_metric_func
from riid.models.base import PyRIIDModel
from riid.models.layers import L1NormLayer


class LabelProportionEstimator(PyRIIDModel):
    """Regressor for predicting label proportions that uses a semi-supervised loss.

    Optionally, a U-spline-based out-of-distribution detection model can be fit to target a desired
    false positive rate.
    """
    UNSUPERVISED_LOSS_FUNCS = {
        "poisson_nll": poisson_nll_diff,
        "normal_nll": normal_nll_diff,
        "sse": sse_diff,
        "weighted_sse": weighted_sse_diff,
        "jsd": jensen_shannon_divergence,
        "chi_squared": chi_squared_diff
    }
    SUPERVISED_LOSS_FUNCS = {
        "sparsemax": (
            SparsemaxLoss,
            {
                "from_logits": True,
                "reduction": tf.keras.losses.Reduction.NONE,
            },
            sparsemax,
        ),
        "categorical_crossentropy": (
            CategoricalCrossentropy,
            {
                "from_logits": True,
                "reduction": tf.keras.losses.Reduction.NONE,
            },
            softmax,
        ),
        "mse": (
            MeanSquaredError,
            {
                "reduction": tf.keras.losses.Reduction.NONE,
            },
            sigmoid,
        )
    }
    INFO_KEYS = (
        # model architecture
        "hidden_layers",
        "learning_rate",
        "epsilon",
        "sup_loss",
        "unsup_loss",
        "metrics",
        "beta",
        "hidden_layer_activation",
        "kernel_l1_regularization",
        "kernel_l2_regularization",
        "bias_l1_regularization",
        "bias_l2_regularization",
        "activity_l1_regularization",
        "activity_l2_regularization",
        "dropout",
        "ood_fp_rate",
        "fit_spline",
        "spline_bins",
        "spline_k",
        "spline_s",
        # dictionaries
        "source_dict",
        # populated when loading model
        "spline_snrs",
        "spline_recon_errors",
    )

    def __init__(self, hidden_layers: tuple = (256,), sup_loss="sparsemax", unsup_loss="sse",
                 metrics: list = ["mae", "categorical_crossentropy"], beta=0.9, source_dict=None,
                 optimizer="adam", optimizer_kwargs=None, learning_rate: float = 1e-3,
                 hidden_layer_activation: str = "mish",
                 kernel_l1_regularization: float = 0.0, kernel_l2_regularization: float = 0.0,
                 bias_l1_regularization: float = 0.0, bias_l2_regularization: float = 0.0,
                 activity_l1_regularization: float = 0.0, activity_l2_regularization: float = 0.0,
                 dropout: float = 0.0, ood_fp_rate: float = 0.05,
                 fit_spline: bool = True, spline_bins: int = 15, spline_k: int = 3,
                 spline_s: int = 0, spline_snrs=None, spline_recon_errors=None):
        """
        Args:
            hidden_layers: tuple defining the number and size of dense layers
            sup_loss: supervised loss function to use for training
            unsup_loss: unsupervised loss function to use for training the
                foreground branch of the network (options: "sse", "poisson_nll",
                "normal_nll", "weighted_sse", "jsd", or "chi_squared")
            metrics: list of metrics to be evaluating during training
            beta: tradeoff parameter between the supervised and unsupervised foreground loss
            source_dict: 2D array of pure, long-collect foreground spectra
            optimizer: tensorflow optimizer or optimizer name to use for training
            optimizer_kwargs: kwargs for optimizer
            learning_rate: learning rate for the optimizer
            hidden_layer_activation: activation function to use for each dense layer
            kernel_l1_regularization: l1 regularization value for the kernel regularizer
            kernel_l2_regularization: l2 regularization value for the kernel regularizer
            bias_l1_regularization: l1 regularization value for the bias regularizer
            bias_l2_regularization: l2 regularization value for the bias regularizer
            activity_l1_regularization: l1 regularization value for the activity regularizer
            activity_l2_regularization: l2 regularization value for the activity regularizer
            dropout: amount of dropout to apply to each dense layer
            ood_fp_rate: false positive rate used to determine threshold for
                out-of-distribution (OOD) detection
            fit_spline: whether or not to fit UnivariateSpline for OOD threshold function
            spline_bins: number of bins used when fitting the UnivariateSpline threshold
                function for OOD detection
            spline_k: degree of smoothing for the UnivariateSpline
            spline_s: positive smoothing factor used to choose the number of knots in the
                UnivariateSpline (s=0 forces the spline through all the datapoints, equivalent to
                InterpolatedUnivariateSpline)
            spline_snrs: SNRs from training used as the x-values to fit the UnivariateSpline
            spline_recon_errors: reconstruction errors from training used as the y-values to
                fit the UnivariateSpline
        """
        super().__init__()

        self.hidden_layers = hidden_layers
        self.sup_loss = sup_loss
        self.unsup_loss = unsup_loss
        self.sup_loss_func, self.activation = self._get_sup_loss_func(
            sup_loss,
            prefix="sup"
        )
        self.sup_loss_func_name = self.sup_loss_func.name

        self.optimizer = optimizer
        if isinstance(optimizer, str):
            self.optimizer = keras.optimizers.get(optimizer)
        if optimizer_kwargs is not None:
            for key, value in optimizer_kwargs.items():
                setattr(self.optimizer, key, value)
        self.optimizer.learning_rate = learning_rate

        self.unsup_loss_func = self._get_unsup_loss_func(unsup_loss)
        self.unsup_loss_func_name = f"unsup_{unsup_loss}_loss"
        self.metrics = metrics
        self.beta = beta
        self.source_dict = source_dict
        self.semisup_loss_func_name = "semisup_loss"
        self.hidden_layer_activation = hidden_layer_activation
        self.kernel_l1_regularization = kernel_l1_regularization
        self.kernel_l2_regularization = kernel_l2_regularization
        self.bias_l1_regularization = bias_l1_regularization
        self.bias_l2_regularization = bias_l2_regularization
        self.activity_l1_regularization = activity_l1_regularization
        self.activity_l2_regularization = activity_l2_regularization
        self.dropout = dropout
        self.ood_fp_rate = ood_fp_rate
        self.fit_spline = fit_spline
        self.spline_bins = spline_bins
        self.spline_k = spline_k
        self.spline_s = spline_s
        self.spline_snrs = spline_snrs
        self.spline_recon_errors = spline_recon_errors
        self.model = None

        self._update_custom_objects("L1NormLayer", L1NormLayer)

    @property
    def source_dict(self) -> dict:
        return self.info["source_dict"]

    @source_dict.setter
    def source_dict(self, value: dict):
        self.info["source_dict"] = value

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

    def _initialize_model(self, input_size, output_size):
        spectra_input = Input(input_size, name="input_spectrum")
        spectra_norm = L1NormLayer(name="normalized_input_spectrum")(spectra_input)
        x = spectra_norm
        for layer, nodes in enumerate(self.hidden_layers):
            x = Dense(
                nodes,
                activation=self.hidden_layer_activation,
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
                name=f"dense_{layer}"
            )(x)

            if self.dropout > 0:
                x = Dropout(self.dropout)(x)
        output = Dense(
            output_size,
            activation="linear",
            name="output"
        )(x)

        self.model = Model(inputs=[spectra_input], outputs=[output])

    def _get_info_as_dict(self):
        info_dict = {}
        for k, v in vars(self).items():
            if k not in self.INFO_KEYS:
                continue
            if isinstance(v, np.ndarray):
                info_dict[k] = v.tolist()
            else:
                info_dict[k] = v
        return info_dict

    def _get_spline_threshold_func(self):
        return UnivariateSpline(
            self.info["avg_snrs"],
            self.info["thresholds"],
            k=self.spline_k,
            s=self.spline_s
        )

    def _fit_spline_threshold_func(self):
        out = pd.qcut(
            np.array(self.spline_snrs),
            self.spline_bins,
            labels=False,
        )
        thresholds = [
            np.quantile(np.array(self.spline_recon_errors)[out == int(i)], 1-self.ood_fp_rate)
            for i in range(self.spline_bins)
        ]
        avg_snrs = [
            np.mean(np.array(self.spline_snrs)[out == int(i)]) for i in range(self.spline_bins)
        ]
        self._update_info(
            avg_snrs=avg_snrs,
            thresholds=thresholds,
            spline_k=self.spline_k,
            spline_s=self.spline_s,
        )

    def _get_snrs(self, ss: SampleSet, bg_cps: float, is_gross: bool) -> np.ndarray:
        fg_counts = ss.info.total_counts.values.astype("float64")
        bg_counts = ss.info.live_time.values * bg_cps
        if is_gross:
            fg_counts = fg_counts - bg_counts
        snrs = fg_counts / np.sqrt(bg_counts)
        return snrs

    def fit(self, seeds_ss: SampleSet, ss: SampleSet, bg_cps: int = 300, is_gross: bool = False,
            batch_size: int = 10, epochs: int = 20, validation_split: float = 0.2,
            callbacks=None, patience: int = 15, es_monitor: str = "val_loss",
            es_mode: str = "min", es_verbose=0, es_min_delta: float = 0.0,
            normalize_sup_loss: bool = True, normalize_func=tf.math.tanh,
            normalize_scaler: float = 1.0, target_level="Isotope", verbose: bool = False):
        """Fit a model to the given SampleSet(s).

        Args:
            seeds_ss: `SampleSet` of pure, long-collect spectra
            ss: `SampleSet` of `n` gross or foreground spectra where `n` >= 1
            bg_cps: background rate assumption used for calculating SNR in spline function
                using in OOD detection
            is_gross: whether `ss` contains gross spectra
            batch_size: number of samples per gradient update
            epochs: maximum number of training iterations
            validation_split: proportion of training data to use as validation data
            callbacks: list of callbacks to be passed to TensorFlow Model.fit() method
            patience: number of epochs to wait for `EarlyStopping` object
            es_monitor: quantity to be monitored for `EarlyStopping` object
            es_mode: mode for `EarlyStopping` object
            es_verbose: verbosity level for `EarlyStopping` object
            es_min_delta: minimum change to count as an improvement for early stopping
            normalize_sup_loss: whether to normalize the supervised loss term
            normalize_func: normalization function used for supervised loss term
            normalize_scaler: scalar that sets the steepness of the normalization function
            target_level: source level to target for model output
            verbose: whether model training output is printed to the terminal
        """
        spectra = ss.get_samples().astype(float)
        sources_df = ss.sources.T.groupby(target_level, sort=False).sum().T
        sources = sources_df.values.astype(float)
        self.sources_columns = sources_df.columns

        if verbose:
            print("Building dictionary...")

        if self.source_dict is None:
            self.source_dict = _get_reordered_spectra(
                seeds_ss.spectra,
                seeds_ss.sources,
                self.sources_columns,
                target_level=target_level
            ).values

        if not self.model:
            if verbose:
                print("Initializing model...")
            self._initialize_model(
                (ss.n_channels,),
                sources.shape[1],
            )
        elif verbose:
            print("Model already initialized.")

        if verbose:
            print("Building loss functions...")

        self.semisup_loss_func = build_keras_semisupervised_loss_func(
            self.sup_loss_func,
            self.unsup_loss_func,
            self.source_dict,
            self.beta,
            self.activation,
            n_labels=sources.shape[1],
            normalize=normalize_sup_loss,
            normalize_func=normalize_func,
            normalize_scaler=normalize_scaler
        )

        semisup_metrics = None
        if self.metrics:
            if verbose:
                print("Building metric functions...")
            semisup_metrics = []
            for each in self.metrics:
                if isinstance(each, str):
                    semisup_metrics.append(
                        build_keras_semisupervised_metric_func(
                            tf.keras.metrics.get(each),
                            self.activation,
                            sources.shape[1]
                        )
                    )
                else:
                    semisup_metrics.append(
                        build_keras_semisupervised_metric_func(
                            each,
                            self.activation,
                            sources.shape[1]
                        )
                    )

        self.model.compile(
            loss=self.semisup_loss_func,
            optimizer=self.optimizer,
            metrics=semisup_metrics
        )

        es = EarlyStopping(
            monitor=es_monitor,
            patience=patience,
            verbose=es_verbose,
            restore_best_weights=True,
            mode=es_mode,
            min_delta=es_min_delta,
        )

        if callbacks:
            callbacks.append(es)
        else:
            callbacks = [es]

        history = self.model.fit(
            spectra,
            np.append(sources, spectra, axis=1),
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split,
            callbacks=callbacks,
            shuffle=True,
            batch_size=batch_size
        )

        if self.fit_spline:
            if verbose:
                print("Finding OOD detection threshold function...")

            train_logits = self.model.predict(spectra, verbose=0)
            train_lpes = self.activation(tf.convert_to_tensor(train_logits, dtype=tf.float32))
            self.spline_recon_errors = reconstruction_error(
                tf.convert_to_tensor(spectra, dtype=tf.float32),
                train_lpes,
                self.source_dict,
                self.unsup_loss_func
            ).numpy()
            self.spline_snrs = self._get_snrs(ss, bg_cps, is_gross)
            self._fit_spline_threshold_func()

        info = self._get_info_as_dict()
        self._update_info(
            target_level=target_level,
            model_outputs=sources_df.columns.values.tolist(),
            normalization=ss.spectra_state,
            **info,
        )

        return history

    def predict(self, ss: SampleSet, bg_cps: int = 300, is_gross: bool = False, verbose=False):
        """Estimate the proportions of counts present in each sample of the provided SampleSet.

        Results are stored inside the SampleSet's prediction_probas property.

        Args:
            ss: `SampleSet` of `n` foreground or gross spectra where `n` >= 1
            bg_cps: background rate used for estimating sample SNRs.
                If background rate varies to a significant degree, split up sampleset
                by SNR and make multiple calls to this method.
            is_gross: whether `ss` contains gross spectra
        """
        test_spectra = ss.get_samples().astype(float)

        logits = self.model.predict(test_spectra, verbose=verbose)
        lpes = self.activation(tf.convert_to_tensor(logits, dtype=tf.float32))

        col_level_idx = SampleSet.SOURCES_MULTI_INDEX_NAMES.index(self.target_level)
        col_level_subset = SampleSet.SOURCES_MULTI_INDEX_NAMES[:col_level_idx+1]
        ss.prediction_probas = pd.DataFrame(
            data=lpes,
            columns=pd.MultiIndex.from_tuples(
               self.get_model_outputs_as_label_tuples(),
               names=col_level_subset
            )
        )

        # Fill in unsupervised losses
        recon_errors = reconstruction_error(
            tf.convert_to_tensor(test_spectra, dtype=tf.float32),
            lpes,
            self.source_dict,
            self.unsup_loss_func
        ).numpy()

        if self.fit_spline:
            snrs = self._get_snrs(ss, bg_cps, is_gross)
            thresholds = self._get_spline_threshold_func()(snrs)
            is_ood = recon_errors > thresholds
            ss.info["ood"] = is_ood

        ss.info["recon_error"] = recon_errors


def _get_reordered_spectra(old_spectra_df: pd.DataFrame, old_sources_df: pd.DataFrame,
                           new_sources_columns, target_level) -> pd.DataFrame:
    collapsed_sources_df = old_sources_df\
        .T.groupby(target_level)\
        .sum().T
    reordered_spectra_df = old_spectra_df.iloc[
        collapsed_sources_df[
            new_sources_columns
        ].idxmax()
    ].reset_index(drop=True)

    return reordered_spectra_df
