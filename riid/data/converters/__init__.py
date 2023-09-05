# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains utilities for converting known datasets into `SampleSet`s."""
import glob
from pathlib import Path
from typing import Callable

import parmap as pm


def _validate_and_create_output_dir(output_dir: str):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)
    if not output_dir_path.is_dir:
        raise ValueError("`output_dir` already exists but is not a directory.")


def convert_directory(input_dir_path: str, conversion_func: Callable, file_ext: str,
                      pm_processes: int = 8, pm_chunksize: int = 1, **kwargs):
    """Convert and save every file in a specified directory in parallel.

    Conversion functions can be found in sub-modules:

    - AIPT: `riid.data.converters.aipt.convert_and_save()`
    - TopCoder: `riid.data.converters.topcoder.convert_and_save()`

    Due to usage of parallel processing, be sure to run this function as follows:

    ```python
    if __name__ == "__main__":
        convert_directory(...)
    ```

    Consider setting `pm_processes` to `multiprocessing.cpu_count()`;
    unfortunately, `pm_chunksize` requires some experimentation to fully optimize.

    Args:
        input_dir_path: directory path containing the input files
        conversion_func: function used to convert a data file to a `SampleSet`
        file_ext: file extension to read in for conversion
        pm_processes: parmap parameter to set the # of processes
        pm_chunksize: parmap parameter to set the chunksize
        kwargs: keyword args passed to underlying conversion_func operations
    """
    input_path = Path(input_dir_path)
    if not input_path.exists() or not input_path.is_dir():
        print(f"No directory at provided input path: '{input_dir_path}'")
        return

    input_file_paths = sorted(glob.glob(f"{input_dir_path}/*.{file_ext}"))

    x = pm.map(
        conversion_func,
        input_file_paths,
        **kwargs,
        pm_processes=pm_processes,
        pm_chunksize=pm_chunksize,
        pm_parallel=True,
        pm_pbar=True,
    )
    return x
