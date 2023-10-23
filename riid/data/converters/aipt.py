# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module provides tools for handling data related to the multi-lab Algorithm Improvement
Program Team (AIPT).
"""
import os
from pathlib import Path
from typing import List

import pandas as pd

from riid import SAMPLESET_HDF_FILE_EXTENSION
from riid.data.converters import _validate_and_create_output_dir
from riid.data.sampleset import SampleSet

ELEMENT_IDS_PER_FILE = [0, 1, 2, 3]
DEFAULT_ECAL = [
    -3.00000,
    3010.00000,
    150.00000,
    0.00000,
    0.00000,
]


def _element_to_ss(data_df: pd.DataFrame, eid: int, description) -> SampleSet:
    if eid not in ELEMENT_IDS_PER_FILE:
        msg = (
            f"Element #{eid} is invalid. "
            "The available options are: {ELEMENT_IDS_PER_FILE}"
        )
        raise ValueError(msg)

    ss = SampleSet()
    ss.spectra = data_df[f"spectrum-channels{eid}"]\
        .str\
        .split(",", expand=True)\
        .astype(int)
    ss.info.live_time = data_df[f"spectrum-lt{eid}"] / 1000
    ss.info.real_time = data_df[f"spectrum-rt{eid}"] / 1000
    ss.info.total_counts = data_df[f"gc{eid}"]
    ss.info.neutron_counts = data_df["nc0"]
    ss.info.ecal_order_0 = DEFAULT_ECAL[0]
    ss.info.ecal_order_1 = DEFAULT_ECAL[1]
    ss.info.ecal_order_2 = DEFAULT_ECAL[2]
    ss.info.ecal_order_3 = DEFAULT_ECAL[3]
    ss.info.ecal_low_e = DEFAULT_ECAL[4]
    ss.info.timestamp = data_df["utc-time"]
    ss.info.description = description
    ss.info["latitude"] = data_df["latitude"]
    ss.info["longitude"] = data_df["longitude"]
    ss.info["is_in_zone"] = data_df["is-in-zone"]
    ss.info["is_closest_approach"] = data_df["is-closest-approach"]
    ss.info["is_source_present"] = data_df["is-source-present"]

    detector_name = f"{data_df['detector'].unique()[0]}.{eid}"
    ss.detector_info = {
        "name": detector_name
    }

    return ss


def aipt_file_to_ss_list(file_path: str) -> List[SampleSet]:
    """Process an AIPT CSV file into a list of SampleSets.

    Each file contains a series of spectra for multiple detectors running simultaneously.
    As such, each `SampleSet` in the list returned by this function represents the data
    collected by each detector.
    Each row of each `SampleSet` represents a measurement from each detector at the same
    moment in time.
    As such, after calling this function, you might consider summing all four spectra at
    each timestep into a single spectrum as another processing step.

    Args:
        file_path: file path of the CSV file

    Returns:
        List of `SampleSet`s each containing a series of spectra for a single run
    """
    data_df = pd.read_csv(file_path, header=0, sep="\t")
    base_description = os.path.splitext(os.path.basename(file_path))[0]

    ss_list = []
    for eid in ELEMENT_IDS_PER_FILE:
        description = f"{base_description}_{eid}"
        ss = _element_to_ss(data_df, eid, description)
        ss_list.append(ss)

    return ss_list


def convert_and_save(input_file_path: str, output_dir: str = None,
                     skip_existing: bool = True, **kwargs):
    """Convert AIPT file to SampleSet and save as HDF.

    Output file will have same name but appended with a detector identifier
    and having a different extension.

    Args:
        input_file_path: file path of the CSV file
        output_dir: alternative directory in which to save HDF files
            (defaults to `input_file_path` parent if not provided)
        skip_existing: whether to skip conversion if the file already exists
        kwargs: keyword args passed to `aipt_file_to_ss_list()` (not currently used)
    """
    input_path = Path(input_file_path)
    if not output_dir:
        output_dir = input_path.parent
    _validate_and_create_output_dir(output_dir)
    output_file_paths = [
        os.path.join(output_dir, input_path.stem + f"-{i}{SAMPLESET_HDF_FILE_EXTENSION}")
        for i in ELEMENT_IDS_PER_FILE
    ]
    all_output_files_exist = all([os.path.exists(p) for p in output_file_paths])
    if skip_existing and all_output_files_exist:
        return

    ss_list = aipt_file_to_ss_list(input_file_path, **kwargs)
    for output_file_path, ss in zip(output_file_paths, ss_list):
        ss.to_hdf(output_file_path)
