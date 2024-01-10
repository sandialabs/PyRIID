# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains multi-layer perceptron classifiers and regressors."""
from typing import Any, List

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import L1L2, l1, l2
from scipy.interpolate import UnivariateSpline

from riid.data.sampleset import SampleSet
from riid.losses import (build_keras_semisupervised_loss_func,
                         chi_squared_diff, jensen_shannon_divergence,
                         normal_nll_diff, poisson_nll_diff,
                         reconstruction_error, sse_diff, weighted_sse_diff)
from riid.losses.sparsemax import SparsemaxLoss, sparsemax
from riid.metrics import (build_keras_semisupervised_metric_func, multi_f1,
                          single_f1)
from riid.models import ModelInput, PyRIIDModel


class MLPClassifier(PyRIIDModel):
    """Multi-layer perceptron classifier."""
    def __init__(self, hidden_layers: tuple = (512,), activation: str = "relu",
                 loss: str = "categorical_crossentropy",
                 optimizer: Any = Adam(learning_rate=0.01, clipnorm=0.001),
                 metrics: tuple = ("accuracy", "categorical_crossentropy", multi_f1, single_f1),
                 l2_alpha: float = 1e-4, activity_regularizer=l1(0), dropout: float = 0.0,
                 learning_rate: float = 0.01, final_activation: str = "softmax"):
        """
        Args:
            hidden_layers: tuple defining the number and size of dense layers
            activation: activate function to use for each dense layer
            loss: loss function to use for training
            optimizer: tensorflow optimizer or optimizer name to use for training
            metrics: list of metrics to be evaluating during training
            l2_alpha: alpha value for the L2 regularization of each dense layer
            activity_regularizer: regularizer function applied each dense layer output
            dropout: amount of dropout to apply to each dense layer
            learning_rate: learning rate to use for an Adam optimizer
        """
        super().__init__()

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.final_activation = final_activation
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
        """Fit a model to the given `SampleSet`(s).

        Args:
            ss: `SampleSet` of `n` spectra where `n` >= 1 and the spectra are either
                foreground (AKA, "net") or gross.
            bg_ss: `SampleSet` of `n` spectra where `n` >= 1 and the spectra are background
            batch_size: number of samples per gradient update
            epochs: maximum number of training iterations
            validation_split: percentage of the training data to use as validation data
            callbacks: list of callbacks to be passed to the TensorFlow `Model.fit()` method
            val_ss: validation set to be used instead of taking a portion of the training data
            val_bg_ss: validation set to be used as background for `val_ss`
            patience: number of epochs to wait for `tf.keras.callbacks.EarlyStopping`
            es_monitor: quantity to be monitored for `tf.keras.callbacks.EarlyStopping`
            es_mode: mode for `tf.keras.callbacks.EarlyStopping`
            es_verbose: verbosity level for `tf.keras.callbacks.EarlyStopping`
            target_level: `SampleSet.sources` column level to use
            verbose: whether to show detailed model training output

        Returns:
            `tf.History` object.

        Raises:
            `ValueError` when no spectra are provided as input
        """
        if ss.n_samples <= 0:
            raise ValueError("No spectr[a|um] provided!")

        x_train = ss.get_samples().astype(float)
        source_contributions_df = ss.sources.groupby(axis=1, level=target_level, sort=False).sum()
        y_train = source_contributions_df.values.astype(float)
        if bg_ss:
            x_bg_train = bg_ss.get_samples().astype(float)

        if val_ss:
            if val_bg_ss:
                val_data = (
                    [val_ss.get_samples().astype(float), val_bg_ss.get_samples().astype(float)],
                    val_ss.get_source_contributions().astype(float),
                )
            else:
                val_data = (
                    val_ss.get_samples().astype(float),
                    val_ss.get_source_contributions().astype(float),
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

            output = Dense(y_train.shape[1], activation=self.final_activation)(x)
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

        # Update model information
        self._update_info(
            target_level=target_level,
            model_outputs=source_contributions_df.columns.values.tolist(),
            normalization=ss.spectra_state,
        )

        return history

    def predict(self, ss: SampleSet, bg_ss: SampleSet = None, verbose=False):
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

        results = self.model.predict(X, verbose=verbose)

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


class MultiEventClassifier(PyRIIDModel):
    """A classifier for spectra from multiple detectors observing the same event."""
    def __init__(self, hidden_layers: tuple = (512,), activation: str = "relu",
                 loss: str = "categorical_crossentropy",
                 optimizer: Any = Adam(learning_rate=0.01, clipnorm=0.001),
                 metrics: list = ["accuracy", "categorical_crossentropy", multi_f1, single_f1],
                 l2_alpha: float = 1e-4, activity_regularizer: tf.keras.regularizers = l1(0),
                 dropout: float = 0.0, learning_rate: float = 0.01):
        """
        Args:
            hidden_layers: tuple containing the number and size of dense layers
            activation: activate function to use for each dense layer
            loss: string name of the loss function to use for training
            optimizer: string name of the optimizer to use for training
            metrics: list of metrics to be evaluating during training
            l2_alpha: alpha value for the L2 regularization of each dense layer
            activity_regularizer: regularizer function applied each dense layer output
            dropout: amount of dropout to apply to each dense layer
            learning_rate: learning rate to use for an Adam optimizer
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
        """Fit a model to the given SampleSet(s).

        Args:
            list_of_ss: list of `SampleSet`s which have prediction_probas populated from
                single-event classifiers
            target_contributions: DataFrame of the contributions for each
                observation. Column titles are the desired label strings.
            batch_size: number of samples per gradient update
            epochs: maximum number of training iterations
            validation_split: percentage of the training data to use as validation data
            callbacks: list of callbacks to be passed to TensorFlow Model.fit() method
            val_model_ss_list: validation set to be used instead of taking a portion of the
                training data
            val_model_target_contributions: target contributions to the model for each sample
            patience: number of epochs to wait for `tf.keras.callbacks.EarlyStopping` object
            es_monitor: quantity to be monitored for `tf.keras.callbacks.EarlyStopping` object
            es_mode: mode for `tf.keras.callbacks.EarlyStopping` object
            es_verbose: verbosity level for `tf.keras.callbacks.EarlyStopping` object
            target_level: source level to target for model output
            verbose: whether to show detailed training output

        Returns:
            `tf.History` object

        Raises:
            `ValueError` when no predictions are provided with `list_of_ss` input
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
        self._update_info(
            target_level=target_level,
            model_outputs=target_contributions.columns.values.tolist(),
            model_inputs=tuple(
                [(ss.classified_by, ss.prediction_probas.shape[1]) for ss in list_of_ss]
            ),
            normalization=tuple(
                [(ss.classified_by, ss.spectra_state) for ss in list_of_ss]
            ),
        )

        return history

    def predict(self, list_of_ss: List[SampleSet], verbose=False) -> pd.DataFrame:
        """Classify the spectra in the provided `SampleSet`(s) based on each one's results.

        Args:
            list_of_ss: list of `SampleSet`s which had predictions made by single-event models

        Returns:
            `DataFrame` of predicted results for the `Sampleset`(s)
        """
        X = [ss.prediction_probas for ss in list_of_ss]
        results = self.model.predict(X, verbose=verbose)

        col_level_idx = SampleSet.SOURCES_MULTI_INDEX_NAMES.index(self.target_level)
        col_level_subset = SampleSet.SOURCES_MULTI_INDEX_NAMES[:col_level_idx+1]
        results_df = pd.DataFrame(
            data=results,
            columns=pd.MultiIndex.from_tuples(
               self.get_model_outputs_as_label_tuples(),
               names=col_level_subset
            )
        )
        return results_df


class LabelProportionEstimator(PyRIIDModel):
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
            tf.keras.losses.CategoricalCrossentropy,
            {
                "from_logits": True,
                "reduction": tf.keras.losses.Reduction.NONE,
            },
            tf.keras.activations.softmax,
        ),
        "mse": (
            tf.keras.losses.MeanSquaredError,
            {
                "reduction": tf.keras.losses.Reduction.NONE,
            },
            tf.keras.activations.sigmoid,
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
                 metrics=("mae", "categorical_crossentropy",), beta=0.9, source_dict=None,
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
            self.optimizer = tf.keras.optimizers.get(optimizer)
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
        spectra_input = tf.keras.layers.Input(input_size, name="input_spectrum")
        spectra_norm = L1NormLayer(name="normalized_input_spectrum")(spectra_input)
        x = spectra_norm
        for layer, nodes in enumerate(self.hidden_layers):
            x = tf.keras.layers.Dense(
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
            patience: number of epochs to wait for tf.keras.callbacks.EarlyStopping object
            es_monitor: quantity to be monitored for tf.keras.callbacks.EarlyStopping object
            es_mode: mode for tf.keras.callbacks.EarlyStopping object
            es_verbose: verbosity level for tf.keras.callbacks.EarlyStopping object
            es_min_delta: minimum change to count as an improvement for early stopping
            normalize_sup_loss: whether to normalize the supervised loss term
            normalize_func: normalization function used for supervised loss term
            normalize_scaler: scalar that sets the steepness of the normalization function
            target_level: source level to target for model output
            verbose: whether model training output is printed to the terminal
        """
        spectra = ss.get_samples().astype(float)
        sources_df = ss.sources.groupby(axis=1, level=target_level, sort=False).sum()
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
                ss.n_channels,
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
                        build_keras_semisupervised_loss_func(
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

            train_logits = self.model.predict(spectra)
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
        .groupby(axis=1, level=target_level)\
        .sum()
    reordered_spectra_df = old_spectra_df.iloc[
        collapsed_sources_df[
            new_sources_columns
        ].idxmax()
    ].reset_index(drop=True)

    return reordered_spectra_df


class L1NormLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        sums = tf.reduce_sum(inputs, axis=-1)
        l1_norm = inputs / tf.reshape(sums, (-1, 1))
        return l1_norm
