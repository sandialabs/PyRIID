# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""
.. include:: ../README.md
"""
import logging
import os
import sys

from pkg_resources import get_distribution

HANDLER = logging.StreamHandler(sys.stdout)
logging.root.addHandler(HANDLER)
logging.root.setLevel(logging.DEBUG)
MPL_LOGGER = logging.getLogger("matplotlib")
MPL_LOGGER.setLevel(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

SAMPLESET_FILE_EXTENSION = ".h5"
PCF_FILE_EXTENSION = ".pcf"
RIID = "riid"

__version__ = get_distribution(RIID).version

__pdoc__ = {
    "riid.data.synthetic.seed.SeedMixer.__call__": True
}
