# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains the base TFModel class."""
import json
import os
import uuid
import warnings
from enum import Enum

import numpy as np
import onnxruntime
import pandas as pd
import tensorflow as tf
import tf2onnx

import riid
from riid.data.labeling import label_to_index_element
from riid.data.sampleset import SampleSet, SpectraState
from riid.metrics import multi_f1, single_f1


class ModelInput(Enum):
    """Enumerates the potential input sources for a model."""
    GrossSpectrum = 0
    BackgroundSpectrum = 1
    ForegroundSpectrum = 2


class PyRIIDModel:
    """Base class for TensorFlow models."""

    CUSTOM_OBJECTS = {"multi_f1": multi_f1, "single_f1": single_f1}
    SUPPORTED_SAVE_EXTS = {"H5": ".h5", "ONNX": ".onnx"}

    def __init__(self, *args, **kwargs):
        self._info = {}
        self._temp_file_path = "temp_model_file" + riid.SAMPLESET_HDF_FILE_EXTENSION

    @property
    def seeds(self):
        return self._info["seeds"]

    @seeds.setter
    def seeds(self, value):
        self._info["seeds"] = value

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, value):
        self._info = value

    @property
    def target_level(self):
        return self._info["target_level"]

    @target_level.setter
    def target_level(self, value):
        if value in SampleSet.SOURCES_MULTI_INDEX_NAMES:
            self._info["target_level"] = value
        else:
            msg = (
                f"Target level '{value}' is invalid.  "
                f"Acceptable levels: {SampleSet.SOURCES_MULTI_INDEX_NAMES}"
            )
            raise ValueError(msg)

    @property
    def model_inputs(self):
        return self._info["model_inputs"]

    @model_inputs.setter
    def model_inputs(self, value):
        self._info["model_inputs"] = value

    @property
    def model_outputs(self):
        return self._info["model_outputs"]

    @model_outputs.setter
    def model_outputs(self, value):
        n_levels = SampleSet.SOURCES_MULTI_INDEX_NAMES.index(self.target_level) + 1
        if all([len(v) == n_levels for v in value]):
            self._info["model_outputs"] = value
        else:
            self._info["model_outputs"] = [
                label_to_index_element(v, self.target_level) for v in value
            ]

    def to_tflite(self, file_path: str = None, quantize: bool = False):
        """Convert the model to a TFLite model and optionally save or quantize it.

        Args:
            file_path: file path at which to save the model
            quantize: whether to apply quantization

        Returns:
            bytes object representing the model in TFLite form
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_model = converter.convert()
        if file_path:
            open(file_path, "wb").write(tflite_model)
        return tflite_model

    def save(self, file_path: str):
        """Save the model to a file.

        Args:
            file_path: file path at which to save the model, can be either .h5 or
                .onnx format

        Raises:
            `ValueError` when the given file path already exists
        """
        if os.path.exists(file_path):
            raise ValueError("Path already exists.")

        root, ext = os.path.splitext(file_path)
        if ext.lower() not in self.SUPPORTED_SAVE_EXTS.values():
            raise NameError("Model must be an .onnx or .h5 file.")

        warnings.filterwarnings("ignore")

        if ext.lower() == self.SUPPORTED_SAVE_EXTS["H5"]:
            self.model.save(file_path, save_format="h5")
            pd.DataFrame(
                [[v] for v in self.info.values()],
                self.info.keys()
            ).to_hdf(file_path, "_info")
        else:
            model_path = root + self.SUPPORTED_SAVE_EXTS["ONNX"]
            model_info_path = root + "_info.json"

            model_info_df = pd.DataFrame(
                [[v] for v in self.info.values()],
                self.info.keys()
            )
            model_info_df[0].to_json(model_info_path, indent=4)

            tf2onnx.convert.from_keras(
                self.model,
                input_signature=None,
                output_path=model_path
            )

        warnings.resetwarnings()

    def load(self, file_path: str):
        """Load the model from a file.

        Args:
            file_path: file path from which to load the model, must be either an
            .h5 or .onnx file
        """

        root, ext = os.path.splitext(file_path)
        if ext.lower() not in self.SUPPORTED_SAVE_EXTS.values():
            raise NameError("Model must be an .onnx or .h5 file.")

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        if ext.lower() == self.SUPPORTED_SAVE_EXTS["H5"]:
            self.model = tf.keras.models.load_model(
                file_path,
                custom_objects=self.CUSTOM_OBJECTS
            )
            self._info = pd.read_hdf(file_path, "_info")[0].to_dict()
        else:
            model_path = root + self.SUPPORTED_SAVE_EXTS["ONNX"]
            model_info_path = root + "_info.json"
            with open(model_info_path) as fin:
                model_info = json.load(fin)
                self._info = model_info
            self.onnx_session = onnxruntime.InferenceSession(model_path)

        warnings.resetwarnings()

    def get_predictions(self, x, **kwargs):
        if self.model is not None:
            outputs = self.model.predict(x, **kwargs)
        elif self.onnx_session is not None:
            outputs = self.onnx_session.run(
                [self.onnx_session.get_outputs()[0].name],
                {self.onnx_session.get_inputs()[0].name: x.astype(np.float32)}
            )[0]
        else:
            raise ValueError("No model found with which to obtain predictions.")
        return outputs

    def serialize(self) -> bytes:
        """Convert model to a bytes object.

        Returns:
            bytes object representing the model on disk
        """
        self.save(self._temp_file_path)
        try:
            with open(self._temp_file_path, "rb") as f:
                data = f.read()
        finally:
            os.remove(self._temp_file_path)

        return data

    def deserialize(self, stream: bytes):
        """Populate the current model with the given bytes object.

        Args:
            stream: bytes object containing the model information
        """
        try:
            with open(self._temp_file_path, "wb") as f:
                f.write(stream)
            self.load(self._temp_file_path)
        finally:
            os.remove(self._temp_file_path)

    def initialize_info(self):
        """Initialize model information with default values."""
        info = {
            "model_id": str(uuid.uuid4()),
            "model_type": self.__class__.__name__,
            "normalization": SpectraState.Unknown,
            "pyriid_version": riid.__version__,
        }
        self.info.update(info)
