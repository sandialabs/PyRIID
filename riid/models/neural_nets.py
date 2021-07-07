# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This module contains a multi-layer perceptron classifier."""
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.regularizers import l1, l2

from riid.models.metrics import multi_f1, single_f1
from riid.sampleset import SampleSet

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


class MLPClassifier:
    def __init__(self, n_inputs: int, n_classes: int, hidden_layers: tuple = (512,),
                 activation: str = "relu", loss: str = "categorical_crossentropy", optimizer="adam",
                 metrics: list = ["accuracy", "categorical_crossentropy", multi_f1, single_f1],
                 l2_alpha: float = 1e-4, activity_regularizer = l1(0), dropout: float = 0.0,
                 load_model_path: str = None, labels: list = None):
        """Initializes the classifier.

        The model is implemented as a tf.keras.Sequential object.

        Args:
            n_inputs: the size of the input layer.
            n_classes: the number of unique labels.
            hidden_layers: a tuple defining the number and size of dense layers.
            activation: the activate function to use for each dense layer.
            loss: the loss function to use for training.
            optimizer: the optimizer to use for training.
            metrics: list of metrics to be evaluating during training.
            l2_alpha: the alpha value for the L2 regularization of each dense layer.
            activity_regularizer: the regularizer function applied to the output of each dense layer.
            dropout: the amount of dropout to apply to each dense layer.
            load_model_path: loads a model from disk, overriding all other parameters.
            labels: optionally store a friendly form of the labels.

        Returns:
            None

        Raises:
            None
        """
        if not load_model_path:
            self.model = Sequential()
            for layer, nodes in enumerate(hidden_layers):
                if layer == 0:
                    self.model.add(
                        Dense(
                            nodes,
                            input_shape=(n_inputs,),
                            activation=activation,
                            activity_regularizer=activity_regularizer,
                            kernel_regularizer=l2(l2_alpha),
                        )
                    )
                else:
                    self.model.add(
                        Dense(
                            nodes,
                            activation=activation,
                            activity_regularizer=activity_regularizer,
                            kernel_regularizer=l2(l2_alpha),
                        )
                    )
                if dropout > 0:
                    self.model.add(Dropout(dropout))

            self.model.add(Dense(n_classes, activation="softmax"))
            self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            self._classes = None
            self._seeds = None
            self.labels = labels
        else:
            self.load(load_model_path)

        self._temp_file_path = "temp.mdl"

    def fit(self, ss: SampleSet, batch_size: int = 200, epochs: int = 20, validation_split: float = 0.2,
            callbacks=None, val_ss: SampleSet = None, patience=15, es_monitor="val_loss", es_mode="min", es_verbose=0,
            verbose=0):
        """Fits a model to the given SampleSet.

        Args:
            ss: a SampleSet of `n` spectra where `n` >= 1.  The spectra in the
                SampleSet should all be normalized and/or pre-processed in the same way.
            batch_size: the number of samples per gradient update.
            epochs: maximum number of training iterations.
            validation_split: the percentage of the training data to use as validation data.
            callbacks: callbacks list to be passed to TensorFlow Model.fit() method.
            val_ss: manually provided validation set (instead of taking a portion of `ss`).
            patience: the number of epochs to wait for tf.keras.callbacks.EarlyStopping object.
            es_monitor: quantity to be monitored for tf.keras.callbacks.EarlyStopping object.
            es_mode: mode for tf.keras.callbacks.EarlyStopping object.
            es_verbose: verbosity level for tf.keras.callbacks.EarlyStopping object.

        Returns:
            A tf.History object.

        Raises:
            ValueError: Raised when:
                - no spectra are provided.
                - spectra channels do not match the size of the model inpute layer.
        """
        if ss.n_samples <= 0:
            raise ValueError("No spectr[a|um] provided!")
        if ss.n_channels != self.model.input_shape[1]:
            msg = "The binning of all spectra must be compatible with the model's input layer!  "
            msg += "Model input layer size is {}; SampleSet channels is {}.".format(self.model.input_shape[1], ss.n_channels)
            raise ValueError(msg)

        x_train = ss.get_features().astype(float)
        y_train = ss.label_matrix.values.astype(float)
        if val_ss:
            val_data = (
                val_ss.get_features().astype(float),
                val_ss.label_matrix.values.astype(float),
            )
            validation_split = None
        else:
            val_data = None
            row_order = np.random.shuffle(np.arange(x_train.shape[0]))
            # Enforce random validation split through shuffling
            x_train = x_train[row_order][0]
            y_train = y_train[row_order][0]

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
        self.labels = ss.label_matrix_labels
        return history

    def predict(self, ss: SampleSet):
        """Classifies the spectra in the provided SampleSet.
           Results are stored inside the SampleSet's prediction-related properties.

        Args:
            ss: a SampleSet of `n` spectra where `n` >= 1.  The spectra in the
                SampleSet should be normalized and/or pre-processing in the same
                way as spectra on which the model was previously trained.

        Returns:
            None

        Raises:
            None
        """
        x_test = ss.get_features().astype(float)
        results = self.model.predict(x_test)  # output size will be n_samples by n_classes
        ss.prediction_probas = pd.DataFrame(data=results, columns=self.labels)
        ss.predictions = self.labels[results.argmax(axis=1)]

    def evaluate(self, ss: SampleSet, verbose=0):
        """Evaluates the model on the provided SampleSet.

        This function is pretty much a wrapper around the tf.Model.evaluate()

        Args:
            ss: a SampleSet object on which to evaluate model performance

        Returns:
            A dictionary of results where keys are metric names and values
            are the metric values.

        Raises:
            None
        """
        x_test = ss.get_features().astype(float)
        y_true = ss.label_matrix.values.astype(float)
        scores = self.model.evaluate(x_test, y_true, verbose=verbose)
        results = {}
        for metric, score in zip(self.model.metrics_names, scores):
            results.update({metric: score})
        return results

    def to_tflite(self, file_path=None, quantize=False):
        """Converts the model to a TFLite model and
        optionally saves it to a file or quantizes it.

        Args:
            file_path: a string for the file path to which to save the model.
            quantize: a bool specifying whether or not to apply quantization to the TFLite model.

        Returns:
            None

        Raises:
            None
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_model = converter.convert()
        if file_path:
            open(file_path, "wb").write(tflite_model)
        return tflite_model

    def save(self, file_path: str):
        """Saves the model to a file.

        Args:
            file_path: a string representing the file path at which to save the model.

        Returns:
            None

        Raises:
            ValueError: Raised when the given file path already exists.
        """
        if os.path.exists(file_path):
            raise ValueError("Path already exists.")
        self.model.save(file_path, save_format="h5")
        pd.DataFrame(self.labels).to_hdf(file_path, "labels")
        pd.DataFrame(self._classes).to_hdf(file_path, "_classes")
        pd.DataFrame(self._seeds).to_hdf(file_path, "_seeds")

    def load(self, file_path: str):
        """Loads the model from a file.

        Args:
            file_path: a string representing the file path from which to load the model.

        Returns:
            None

        Raises:
            None
        """
        self.model = load_model(
            file_path,
            custom_objects={"multi_f1": multi_f1, "single_f1": single_f1}
        )
        self.labels = pd.read_hdf(file_path, "labels").values.flatten()
        try:
            self._classes = pd.read_hdf(file_path, "_classes").values.flatten()
        except KeyError:
            self._classes = None
        try:
            self._seeds = pd.read_hdf(file_path, "_seeds")
        except KeyError:
            self._seeds = None

    def serialize(self) -> bytes:
        """Converts the model to a bytes object.

        Args:
            None

        Returns:
            Returns a bytes object representing the binary of an HDF file.

        Raises:
            None
        """
        self.save(self._temp_file_path)
        with open(self._temp_file_path, "rb") as f:
            data = f.read()
        os.remove(self._temp_file_path)
        return data

    def deserialize(self, stream: bytes):
        """Populates the current model with the give bytes object.

        Args:
            stream: a bytes object containing the model information.

        Returns:
            None

        Raises:
            None
        """
        with open(self._temp_file_path, "wb") as f:
            f.write(stream)
        self.load(self._temp_file_path)
        os.remove(self._temp_file_path)
