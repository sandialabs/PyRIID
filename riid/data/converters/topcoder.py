# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module provides tools for handling data related to the
"Detecting Radiological Threats in Urban Areas" TopCoder Challenge.
https://doi.org/10.1038/s41597-020-00672-2
"""
import csv
import logging
import os

import numpy as np
import pandas as pd

from riid.data.labeling import label_to_index_element
from riid.data.sampleset import SampleSet

SOURCE_ID_TO_LABEL = {
    0: "Background",
    1: "HEU",
    2: "WGPu",
    3: "I131",
    4: "Co60",
    5: "Tc99m",
    6: "HEU + Tc99m",
}
DISTINCT_SOURCES = list(SOURCE_ID_TO_LABEL.values())


def _get_answers(answer_file_path: str):
    answers = {}
    with open(answer_file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        _ = next(reader)  # skip header
        # timestamp = 0  # in milliseconds
        for row in reader:
            run_id = row[0]
            source_id = int(row[1])
            source_time_secs = float(row[2])
            answers[run_id] = {
                "source_id": source_id,
                "source_time_secs": source_time_secs
            }
    return answers


def topcoder_file_to_ss(file_path: str, sample_interval: float, n_bins: int = 1024,
                        max_energy_kev: int = 3000, answers_path: str = None) -> SampleSet:
    """Convert a TopCoder CSV file of list-mode data into a SampleSet.

    Args:
        file_path: file path of the CSV file
        sample_interval: integration time (referred to as "live time" later) to use in seconds.
            Warning: the final sample in the set will likely be truncated, i.e., the count rate
            will appear low because the live time represented is too large.
            Consider ignoring the last sample.
        n_bins: desired number of bins in the resulting spectra.
            Bins will be uniformly spaced from 0 to `max_energy_kev`.
        max_energy_kev: desired maximum of the energy range represented in the resulting spectra.
            Intuition (assuming a fixed number of bins): a higher max energy value "compresses" the
            spectral information; a lower max energy value spreads out the spectral information and
            counts are potentially lost off the high energy end of the specturm.
        answers_path: path to the answer key for the data. If provided, this will fill out the
            `SampleSet.sources` DataFrame.

    Returns:
        `SampleSet` containing the series of spectra for a single run
    """
    file_name_with_dir = os.path.splitext(file_path)[0]
    file_name = os.path.basename(file_name_with_dir)
    slice_duration_ms = sample_interval * 1000  # in milliseconds
    events = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        timestamp = 0  # in milliseconds
        for row in reader:
            timestamp += int(row[0]) / 1000  # microseconds to milliseconds
            energy = float(row[1])
            if energy > max_energy_kev:
                msg = (
                    f"Encountered energy ({energy:.0f} keV) greater than "
                    f"specified max energy ({max_energy_kev:.0f})"
                )
                logging.warn(msg)
            channel = int(n_bins * energy // max_energy_kev)  # energy to bin
            events.append((timestamp, channel, 1))

    events_df = pd.DataFrame(
        events,
        columns=["timestamp", "channel", "counts"]
    )
    # Organize events into time intervals
    event_time_intervals = pd.cut(
        events_df["timestamp"],
        np.arange(
            start=0,
            stop=events_df["timestamp"].max()+slice_duration_ms,
            step=slice_duration_ms
        )
    )
    # Group events by time intervals
    event_time_groups = events_df.groupby(event_time_intervals)
    # Within time intervals, sum counts by channel
    result = event_time_groups.apply(
        lambda x: x.groupby("channel").sum().loc[:, ["counts"]]
    )
    # Create new dataframe where: row = time interval, column = channel
    spectra_df = result.unstack(level=-1, fill_value=0)
    spectra_df = spectra_df['counts']
    # Add in missing columns as needed
    col_list = np.arange(0, n_bins)
    cols_to_add = np.setdiff1d(col_list, spectra_df.columns)
    missing_df = pd.DataFrame(
        0,
        columns=cols_to_add,
        index=spectra_df.index
    )
    combined_unsorted_df = pd.concat([spectra_df, missing_df], axis=1)
    combined_df = combined_unsorted_df.reindex(
        sorted(combined_unsorted_df.columns),
        axis=1
    )
    spectra_df = combined_df.astype(int)
    spectra_df.reset_index(inplace=True, drop=True)

    # SampleSet creation
    ss = SampleSet()
    ss.measured_or_synthetic = "synthetic"
    ss.detector_info = {
        "name": "2\"x4\"x16\" NaI(Tl)",
        "height_cm": 100,
        "fwhm_at_661_kev": 0.075,
    }
    ss.spectra = spectra_df
    ss.info.total_counts = spectra_df.sum(axis=1)
    ss.info.live_time = sample_interval
    ss.info.description = file_name
    ss.info.ecal_order_0 = 0
    ss.info.ecal_order_1 = max_energy_kev
    ss.info.ecal_order_2 = 0
    ss.info.ecal_order_3 = 0
    ss.info.ecal_low_e = 0

    if answers_path:
        answers = _get_answers(answers_path)
        run_id = file_name.split("runID-")[-1].split(".")[0]
        ss.info.timestamp = answers[run_id]["source_time_secs"]
        source_id = answers[run_id]["source_id"]
        sources_mi = pd.MultiIndex.from_tuples([
            label_to_index_element(x, label_level="Seed")
            for x in DISTINCT_SOURCES
        ], names=SampleSet.SOURCES_MULTI_INDEX_NAMES)
        sources_df = pd.DataFrame(
            np.zeros((ss.n_samples, len(DISTINCT_SOURCES))),
            columns=sources_mi,
        )
        sources_df.sort_index(axis=1, inplace=True)
        source = label_to_index_element(SOURCE_ID_TO_LABEL[source_id], label_level="Seed")
        sources_df[source] = 1.0
        ss.sources = sources_df

    return ss
