# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""
.. include:: ../README.md
"""
import logging
import os
import sys

from importlib.metadata import version

HANDLER = logging.StreamHandler(sys.stdout)
logging.root.addHandler(HANDLER)
logging.root.setLevel(logging.DEBUG)
MPL_LOGGER = logging.getLogger("matplotlib")
MPL_LOGGER.setLevel(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

SAMPLESET_HDF_FILE_EXTENSION = ".h5"
SAMPLESET_JSON_FILE_EXTENSION = ".json"
PCF_FILE_EXTENSION = ".pcf"
ONNX_MODEL_FILE_EXTENSION = ".onnx"
TFLITE_MODEL_FILE_EXTENSION = ".tflite"
RIID = "riid"

__version__ = version(RIID)

__pdoc__ = {
    "riid.data.synthetic.seed.SeedMixer.__call__": True,
    "riid.data.synthetic.passby.PassbySynthesizer._generate_single_passby": True,
    "riid.data.sampleset.SampleSet._channels_to_energies": True,
}
