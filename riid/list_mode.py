# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains utilities for dealing with list mode data."""
import datetime
import os
import pathlib
from typing import Union

import numpy as np
import pandas as pd


def get_list_file_spectra(list_file_path: str, sampling_frequency: float):
    """Returns the time varying histogram of the contents of a list file and the time edges.

    Args:
        list_file_path: Defines the string reprsenting the relative path to the list mode file.
        sampling_frequency: Defines the number of samples per second, or sampling frequency
            in hertz.

    Returns:
        The result of the call to make_time_slice_histograms with the data.

    """
    data = read_list_file(list_file_path)["data"]
    return make_time_slice_histograms(data, sampling_frequency)


def make_time_slice_histograms(data: pd.DataFrame, spectrum_rate_hz: float):
    """Produces histograms from timeslices of the list mode data, where the
    histograms are given as an array of timeslices by channels.

    Args:
        data: Defines a DataFrame of the list mode data, e.g. sample records.
        spectrum_rate_hz: Defines the sampling frequency of the records in hertz.

    Returns:
        A tuple of the form: (2D array which is n_time_slices X n_channels, time_edges).

    """
    max_channel = _next_highest_power_of_two(data.channel.max())
    time_edges = np.arange(0, data.timestamp.max(), 1/spectrum_rate_hz)
    upper_indices = np.searchsorted(data.timestamp, time_edges[1:])
    lower_indices = np.insert(upper_indices[:-1], 0, 0)
    ranges = np.vstack((lower_indices, upper_indices)).T
    all_counts = data.channel.values
    hists = np.zeros([len(ranges), max_channel])
    for i, limits in enumerate(ranges):
        hists[i, :] = np.bincount(all_counts[limits[0]:limits[1]], minlength=max_channel)
    return hists, time_edges


def _next_highest_power_of_two(value: int):
    """Returns the next highest power of two for a given value, e.g.,
    given 938 will return 1024.

    Args:
        value: Defines a value for which the next-highest power of two
            should be calculated.

    Returns:
        An integer representing the first power of two that is greater
            than the provided value.

    """
    p = 1
    lower = True
    while lower:
        nth_power = 2**p
        lower = (value > nth_power)
        p += 1

    return nth_power


def read_list_file(list_file_path: Union[str, os.Path, pathlib.Path]):
    """Reads a list file from the given path.

    Args:
        list_file_path: Defines the string or Path to the file to be read.

    Returns:
        A dict with the keys "header", "times", and "amplitudes".

    """
    data = np.fromfile(list_file_path, dtype="uint8")
    header_length = 256
    header_data = data[:header_length]

    data = np.fromfile(list_file_path, dtype="uint32")
    event_data = data[int(header_length/4):]

    header_data = _interpret_list_mode_header(header_data)
    assert (header_data["data_style"] == 1), "Only digiBASE data format has been implemented. \
            List data style was {}".format(header_data["data_style"])
    # Indicates digiBASE format
    times, amplitudes = _interpret_list_mode_data(event_data)
    data = pd.DataFrame(data=np.array(times/1e6, dtype=np.float64), columns=["timestamp"])
    data["channel"] = amplitudes
    offset = np.array(2**21/1e6, dtype=np.float64)
    jumps = np.logical_or(data.timestamp.diff() > 1e12, data.timestamp.diff() < 0)
    data["timestamp"] += jumps.cumsum() * offset

    return {"header": header_data, "data": data}


def _interpret_list_mode_header(header_bytes: np.array):
    """Reads in list mode header.

    Args:
        header_bytes: Defines the array of bytes containing the header
            information to be read.

    Returns:
        A dictionary with the parsed header data.

    """
    assert (len(header_bytes) == 256), "'header_bytes' must contain exactly 256 bytes"

    hdef = _get_header_definition()

    header_data = {}

    for key in hdef:
        target_format = hdef[key]["d_type"]
        begin, end = hdef[key]["offset"], hdef[key]["offset"] + hdef[key]["n_bytes"]
        bytes_of_interest = header_bytes[begin:end]
        if target_format == np.char:
            value = "".join(np.char.mod("%c", bytes_of_interest))
            if value == "\x01":
                value = ""
        else:
            value = np.frombuffer(bytes_of_interest, dtype=target_format)[0]
        if key == "acquisition_start_time":
            value = _ole_days_to_datetime(value)
        header_data.update({key: value})
    return header_data


def _determine_if_event(four_byte_int: int):
    """Checks if event bit is set.

    Args:
        four_byte_int: Defines a four byte integer mask containing
            the bit to be checked.

    Returns:
        The state of the event bit.

    """
    return four_byte_int >> 31 == 0


def _interpret_list_mode_data(data_bytes: np.array):
    """Parses out bytes into different entries.

    Args:
        data_bytes: Defines the array of bytes to be parsed.

    Returns:
        The amplitudes and times of the list mode data.

    """
    times, amplitudes = _interpret_event_word(data_bytes)
    event_entries = _determine_if_event(data_bytes)
    times = times[event_entries]
    amplitudes = amplitudes[event_entries]

    return times, amplitudes


def _ole_days_to_datetime(float_days: float):
    """Converts timestamps from the Windows/MS base date
    to a python datetime.

    Args:
        float_days: Defines the number of days since
            12/30/1899 12:00 am, the base date.

    Returns:
        A datetime object representing the timestamp.

    """
    t0 = datetime.datetime(1899, 12, 30, 0, 0, 0)
    days = int(np.floor(float_days))
    d_rem = float_days-days
    hours = int(np.floor(d_rem*24))
    h_rem = (d_rem * 24) - hours
    minutes = int(np.floor(h_rem*60))
    m_rem = (h_rem * 60) - minutes
    seconds = int(np.floor(m_rem*60))
    s_rem = (m_rem * 60) - seconds
    micro_seconds = int(s_rem * 1e6)
    dt = t0 + datetime.timedelta(
        days=days,
        seconds=seconds,
        microseconds=micro_seconds,
        hours=hours,
        minutes=minutes
    )
    return dt


def _interpret_event_word(four_byte_int: int):
    """Interprets an event word time, amplitude.
        BIT   | DESCRIPTION
        31    | 0 for event
        30-21 | Amplitude of Pulse
        20-0  | Time in microseconds that the event occured

    Args:
        four_byte_int: Defines the integer mask containing the
            word data.

    Returns:
        A tuple of the time, amplitude data for the word.

    """
    amplitude_mask = 4292870144
    time_mask = 2097151

    amplitude = np.bitwise_and(amplitude_mask, four_byte_int) >> 21
    time = np.bitwise_and(time_mask, four_byte_int, dtype=np.uint64)

    return(time, amplitude)


def _interpret_time_stamp_word(four_byte_int: int):
    """Interprets an event word timestamp.
        BIT  | DESCRIPTION
        31   | 1 for time_only
        30-0 | Current Time in Microseconds

    Args:
        four_byte_int: Defines the integer mask containing the
            word data.

    Returns:
        A tuple of the time, amplitude data for the word.

    """
    time_mask = 2147483647
    return np.bitwise_and(time_mask, four_byte_int)


def _get_header_definition():
    """Returns dictionary of header definition for list mode data.

    Returns:
        A dictionary of list mode data header values.

    """
    definition = {
        "header_format": {
            "offset": 0,
            "n_bytes": 4,
            "d_type": np.int32
        },
        "data_style": {
            "offset": 4,
            "n_bytes": 4,
            "d_type": np.int32
        },
        "acquisition_start_time": {
            "offset": 8,
            "n_bytes": 8,
            "d_type": np.double
        },
        "device_address": {
            "offset": 16,
            "n_bytes": 80,
            "d_type": np.char
        },
        "mcb_type_string": {
            "offset": 96,
            "n_bytes": 9,
            "d_type": np.char
        },
        "device_serial_number": {
            "offset": 105,
            "n_bytes": 16,
            "d_type": np.char
        },
        "data_description": {
            "offset": 121,
            "n_bytes": 80,
            "d_type": np.char
        },
        "energy_calibration_valid": {
            "offset": 201,
            "n_bytes": 1,
            "d_type": np.char
        },
        "energy_units": {
            "offset": 202,
            "n_bytes": 4,
            "d_type": np.float32
        },
        "e_cal_offset": {
            "offset": 206,
            "n_bytes": 4,
            "d_type": np.float32
        },
        "e_cal_linear": {
            "offset": 210,
            "n_bytes": 4,
            "d_type": np.float32
        },
        "e_cal_quadratic": {
            "offset": 214,
            "n_bytes": 4,
            "d_type": np.float32
        },
        "shape_calibration_valid": {
            "offset": 218,
            "n_bytes": 1,
            "d_type": np.char
        },
        "s_cal_offset": {
            "offset": 219,
            "n_bytes": 4,
            "d_type": np.float32
        },
        "s_cal_linear": {
            "offset": 223,
            "n_bytes": 4,
            "d_type": np.float32
        },
        "s_cal_quadratic": {
            "offset": 227,
            "n_bytes": 4,
            "d_type": np.float32
        },
        "conversion_gain": {
            "offset": 231,
            "n_bytes": 4,
            "d_type": np.int32
        },
        "detector_ID_number": {
            "offset": 235,
            "n_bytes": 4,
            "d_type": np.int32
        },
        "real_time": {
            "offset": 239,
            "n_bytes": 4,
            "d_type": np.float32
        },
        "live_time": {
            "offset": 243,
            "n_bytes": 4,
            "d_type": np.float32
        },
        "unused": {
            "offset": 247,
            "n_bytes": 9,
            "d_type": np.char
        },
    }
    return definition
