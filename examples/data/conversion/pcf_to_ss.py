# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to convert PCFs to `SampleSet`s
and save as HDF5 files.

This example shows how the utilities originally built for converting Topcoder and AIPT data
can be easily repurposed to bulk convert any type of file.
PyRIID just so happens to already have PCF reading available (via `read_pcf()`),
but a custom file format is easily handled by implementing your own `convert_and_save()` function.
"""
import os
from pathlib import Path

from riid import SAMPLESET_HDF_FILE_EXTENSION, read_pcf
from riid.data.converters import (_validate_and_create_output_dir,
                                  convert_directory)


def convert_and_save(input_file_path: str, output_dir: str = None,
                     skip_existing: bool = True, **kwargs):
    input_path = Path(input_file_path)
    if not output_dir:
        output_dir = input_path.parent
    _validate_and_create_output_dir(output_dir)
    output_file_path = os.path.join(output_dir, input_path.stem + SAMPLESET_HDF_FILE_EXTENSION)
    if skip_existing and os.path.exists(output_file_path):
        return

    output_file_path = os.path.join(
        input_path.parent,
        input_path.stem + SAMPLESET_HDF_FILE_EXTENSION
    )
    ss = read_pcf(input_file_path)
    ss.to_hdf(output_file_path)


if __name__ == "__main__":
    # Change the following to a valid path on your computer
    DIRECTORY_WITH_FILES_TO_CONVERT = "./data"

    convert_directory(
        DIRECTORY_WITH_FILES_TO_CONVERT,
        convert_and_save,
        file_ext="pcf",
        output_dir=DIRECTORY_WITH_FILES_TO_CONVERT + "/converted",
    )
