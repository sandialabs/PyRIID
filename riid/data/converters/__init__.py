# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains utilities for converting known datasets into `SampleSet`s."""
import glob
import os

import parmap as pm


def convert_directory(input_dir_path, conversion_func, output_dir_path=None,
                      pm_processes=8, pm_chunksize=100, **kwargs):
    """Convert and save every file in a specified directory in parallel.

    Conversion functions can be found in sub-modules:

    - `riid.data.converters.aipt.aipt_file_to_ss()`
    - `riid.data.converters.topcoder.topcoder_file_to_ss()`

    Due to usage of parallel processing, be sure to run this function as follows:

    ```python
    if __name__ == '__main__'
        convert_directory(...)
    ```

    Consider setting `pm_processes` to `multiprocessing.cpu_count()`;
    unfortunately, `pm_chunksize` requires some experimentation to fully optimize.

    Args:
        input_dir_path: directory path containing the input CSVs
        conversion_func: function used to convert a data file to a `SampleSet`
        output_dir_path: directory path in which to save processed files (in HDF)
        pm_processes: parmap parameter to set the # of processes
        pm_chunksize: parmap parameter to set the chunksize
        kwargs: keyword-args passed to process_file()
    """
    input_file_paths = sorted(glob.glob(f"{input_dir_path}/*.csv"))
    if not output_dir_path or not os.path.exists(output_dir_path):
        output_dir_path = input_dir_path
    output_file_paths = [
        os.path.join(
            output_dir_path, os.path.splitext(os.path.basename(x))[0] + ".h5"
        )
        for x in input_file_paths
    ]

    def _convert_and_save_func(input_file_path, output_file_path, **kwargs):
        ss = conversion_func(input_file_path, **kwargs)
        ss.to_hdf(output_file_path)

    args = list(zip(input_file_paths, output_file_paths))
    _ = pm.starmap(
        _convert_and_save_func,
        args,
        **kwargs,
        pm_processes=pm_processes,
        pm_chunksize=pm_chunksize,
        pm_parallel=True,
        pm_pbar=True,
    )
