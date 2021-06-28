# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This module sets up shared logging functionality and global variables."""
import logging
import os
import sys


HANDLER = logging.StreamHandler(sys.stdout)
logging.root.addHandler(HANDLER)
logging.root.setLevel(logging.DEBUG)
MPL_LOGGER = logging.getLogger('matplotlib')
MPL_LOGGER.setLevel(logging.WARNING)

SMPL_FILE_EXTENSION = ".smpl"
PYRIID_DATA_DIR_ENV_KEY = "PYRIID_DATA_DIR"
DATA_PATH = ""
# Check for required environment variables relating to data directory
if PYRIID_DATA_DIR_ENV_KEY in os.environ:
    DATA_PATH = os.environ[PYRIID_DATA_DIR_ENV_KEY]
else:
    msg = "The following '{}' environment variable is not defined.  ".format(
        PYRIID_DATA_DIR_ENV_KEY
    )
    msg += "Note that certain, optional functions will not work."
    logging.warn(msg)

DATA_PATH = os.path.expanduser(DATA_PATH)


class DataDirectoryNotFoundError(Exception):
    def __init__(self, *args, **kwargs):
        default_message = "The path to the data directory does not exist."
        if not (args or kwargs):
            args = (default_message,)
        super().__init__(*args, **kwargs)
