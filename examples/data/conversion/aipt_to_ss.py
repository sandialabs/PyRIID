# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to convert AIPT data files to `SampleSet`s
and save as HDF5 files.
"""
from riid.data.converters import convert_directory
from riid.data.converters.aipt import convert_and_save


if __name__ == "__main__":
    # Change the following to a valid path on your computer
    DIRECTORY_WITH_FILES_TO_CONVERT = "./data"

    convert_directory(
        DIRECTORY_WITH_FILES_TO_CONVERT,
        convert_and_save,
        file_ext="open",
        output_dir=DIRECTORY_WITH_FILES_TO_CONVERT + "/converted",
    )
