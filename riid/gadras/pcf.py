# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains utilities for working with GADRAS PCF files."""
import re
import struct
from collections import defaultdict

import numpy as np
import pandas as pd

from riid.data import SampleSet
from riid.data.labeling import (NO_CATEGORY, NO_ISOTOPE, NO_SEED,
                                _find_category, _find_isotope)


HEADER_DEFINITIONS = defaultdict(lambda: {
    "fields": (
        "NRPS",
        "Version",
        "Ecal_label",
        "Energy_Calibration_Offset",
        "Energy_Calibration_Gain",
        "Energy_Calibration_Quadratic",
        "Energy_Calibration_Cubic",
        "Energy_Calibration_Low_Energy"
    ),
    "mask": "<h3s4cfffff",
    "n_bytes": [2, 3, 4, 4, 4, 4, 4, 4]
})

HEADER_DEFINITIONS["DHS"] = {
    "fields": (
        "NRPS",
        "Version",
        "last_mod_hash",
        "UUID",
        "Inspection",
        "Lane_Number",
        "Measurement_Remark",
        "Intrument_Type",
        "Manufacturer",
        "Instrument_Model",
        "Instrument_ID",
        "Item_Description",
        "Item_Location",
        "Measurement_Coordinates",
        "Item_to_detector_distance",
        "Occupancy_Number",
        "Cargo_Type"
    ),
    "mask": "<h3s7s36s16sh26s28s28s18s18s20s16s16shh16s",
    "n_bytes": [2, 3, 7, 36, 16, 2, 26, 28, 28, 18, 18, 20, 16, 16, 2, 2, 16]
}

SPECTRUM_DEFINITION = {
    "fields": (
        "Compressed_Text_Buffer",
        "Date-time_VAX",
        "Tag",
        "Live_Time",
        "Total_time_per_real_time",
        "unused0",
        "unused1",
        "unused2",
        "Energy_Calibration_Offset",
        "Energy_Calibration_Gain",
        "Energy_Calibration_Quadratic",
        "Energy_Calibration_Cubic",
        "Energy_Calibration_Low_Energy",
        "Occupancy_Flag",
        "Total_Neutron_Counts",
        "Number_of_Channels"
    ),
    "mask": "180s23scffffffffffffi",
    "n_bytes": [180, 23, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
}


def pcf_to_smpl(pcf_file_path: str, verbose: bool = False):
    """Converts pcf file to sampleset object.

    Args:
        pcf_file_path: Defines the path to the pcf file to be converted.
        verbose: Determines whether or not to show verbose function output in terminal.

    Returns:
        A Sampleset object containing information of pcf file in sampleset format

    Raises:
        None.
    """
    return _pcf_dict_to_sampleset(_pcf_to_dict(pcf_file_path, verbose))


def smpl_to_pcf(ss: SampleSet, save_path: str, verbose: bool = True):
    """Saves a SampleSet to a PCF file.

    Args:
        ss: Defines the Sampleset to convert.
        save_path: Defines the path at which to save the converted Sampleset.
        verbose: Determines whether or not to show verbose function output in terminal.

    Raises:
        None.
    """
    _dict_to_pcf(_smpl_to_dict(ss), save_path, verbose=verbose)


def _get_srsi(file_bytes: list):
    """Obtains the PCF file Spectral Records Start Index (SRSI).

    Args:
        file_bytes: Defines a byte array of the pcf file contents.

    Returns:
        A tuple containing the return_value and srsi.
        return_value: The string indicator for what Deviation pairs type is being used.
        srsi: Spectral Records Start Index.  Index in file where spectra records begin
        after skipping past deviation pairs.

    Raises:
        None.
    """
    test_lengths = [30, 20]
    index = 256
    return_value = "other"
    srsi = 2
    for i in test_lengths:
        value = struct.unpack(
            "{}s".format(i),
            file_bytes[index:index + i]
        )[0].decode("utf-8").strip()
        if value in "DeviationPairsInFile" or value in "DeviationPairsInFileCompressed":
            return_value = value
            srsi = 83
            break

    return return_value, srsi


def _get_spectrum_header_offset(spectrum_number: int, srsi: int, nrps: int):
    """Calculates the PCF header offset for spectrum.

    Args:
        spectrum_number: Defines the number of spectrum for which to the
            obtain offset of header.
        srsi: Defines the spectral record start index.
        nrps: Defines the number of records (channels) per spectrum.

    Returns:
        An integer representing the offset of the spectrum header.

    Raises:
        None.
    """
    return 256*(srsi + nrps * (spectrum_number - 1) - 1)


def _read_header(header_bytes: list, header_def: dict):
    """Converts bytes of header using header definition.

    Args:
        header_bytes: Defines the byte array of the values of the header.
        header_def: Dictionary defining the: mask, field names, and lengths of each entry.

    Returns:
        A dictionary containing the content of the header.

    Raises:
        None.
    """
    header_values = struct.unpack(
        header_def["mask"],
        header_bytes[:sum(header_def["n_bytes"])]
    )
    header = {}
    for field, value in zip(header_def["fields"], header_values):
        if isinstance(value, bytes) and field != "last_mod_hash":
            value = value.decode("utf-8", "ignore")
        header.update({field: value})
    return header


def _read_spectra(data: list, n_rec_per_spec: int, spec_rec_start_indx: int):
    """Reads spectra from PCF file.

    Args:
        data: Defines the byte array of pcf file contents.
        n_rec_per_spec: Defines the number of records per spectrum. (NRPS in pcf ICD).
        spec_rec_start_index: Defines spectral record start index. (SRSI in pcf ICD).

    Returns:
        A list of dictionaries containing pcf file spectra information.

    Raises:
        None.
    """
    num_samples = int((len(data) - 256 * (spec_rec_start_indx - 1)) / (256 * n_rec_per_spec))
    spectra = []
    # The range below is due to the definition of the spectrum offset in
    # pcf definition document assuming a first index of 1
    for spectrum_number in range(1, num_samples + 1):
        spectrum_header_offset = _get_spectrum_header_offset(
            spectrum_number,
            spec_rec_start_indx,
            n_rec_per_spec
        )
        header_def = SPECTRUM_DEFINITION
        spectrum_header = _read_header(
            data[spectrum_header_offset: spectrum_header_offset+256],
            header_def
        )

        spctrum_offset = spectrum_header_offset + 256
        n_channels = int(256*(n_rec_per_spec-1) / 4)
        values = struct.unpack(
            "{}f".format(n_channels),
            data[spctrum_offset: spctrum_offset+(4*n_channels)]
        )

        spectrum = {
            "header": spectrum_header,
            "spectrum": np.array(values)
        }
        spectra.append(spectrum)

    return spectra


def _pcf_to_dict(pcf_file_path: str, verbose: bool = False):
    """Converts .pcf file into a python dictionary.

    Args:
        pcf_file_path: Defintes the path to the pcf to be converted.
        verbose: Determines whether or not to show verbose function output in terminal.

    Returns:
        A dictionary with keys "header" and "spectra".

    Raises:
        None.
    """
    pcf_data = np.fromfile(pcf_file_path, dtype=np.uint8)
    version = struct.unpack("3s", pcf_data[2:5])[0].decode("utf-8")
    header_def = HEADER_DEFINITIONS[version]
    deviation_values = struct.unpack("5120f", pcf_data[512:512+20480])
    if not set(deviation_values) == {0} and verbose:
        print("Deviation values exist in file")
    # Header provides metadata about the collection. Spectra is a list
    #  of "header" and "spectrum" pairs for each spectrum in file.
    header = _read_header(pcf_data[:256], header_def)
    dev_type, spec_rec_start_indx = _get_srsi(pcf_data)
    header.update({"SRSI": spec_rec_start_indx, "DevType": dev_type})
    spectra = _read_spectra(pcf_data, header["NRPS"], spec_rec_start_indx)

    return {"header": header, "spectra": spectra}


def _pcf_dict_to_sampleset(pcf_dict: dict):
    """Converts pcf dictionary into a SampleSet.

    Args:
        pcf_dict: Defines the dictionary of pcf values.

    Returns:
        A Sampleset object containing the pcf dict values.

    Raises:
        None.
    """
    if not pcf_dict["spectra"]:
        return

    num_spectra = len(pcf_dict["spectra"])
    num_channels = len(pcf_dict["spectra"][0]["spectrum"])

    sources = []
    infos = []
    spectra = np.ndarray((num_spectra, num_channels))

    for i in range(0, num_spectra):
        spectrum = pcf_dict["spectra"][i]
        compressed_text_buffer = spectrum["header"]["Compressed_Text_Buffer"]

        title, description, source = _unpack_compressed_text_buffer(compressed_text_buffer)

        if not title and not source:
            source = NO_ISOTOPE
            title = NO_ISOTOPE
            category = NO_CATEGORY
        else:
            if not title:
                title = _find_isotope(source)
            elif not source:
                source = title
                title = _find_isotope(source)
            category = _find_category(title)

        # Extract any additional information from the source string
        distance_search = re.search("@ ([0-9,.]+)", source)
        if distance_search:
            distance = float(distance_search.group(1)) / 100
        else:
            distance = np.nan
        an_finds = re.findall("an=([0-9]+)", source)
        if an_finds:
            an = int(an_finds[0])
        else:
            an = None
        ad_finds = re.findall("ad=([0-9, .]+)", source)
        if ad_finds:
            ad_string = ad_finds[0].replace(",", "")
            ad = float(ad_string)
        else:
            ad = None

        # PCF file contains energy calibration terms which are defined as:
        # E_i = a0 + a1*x + a2*x^2 + a3*x^3 + a4 / (1 + 60*x)
        # where:
        #   a0 = order_0
        #   a1 = order_1
        #   a2 = order_2
        #   a3 = order_3
        #   a4 = low_E
        #   x = channel number
        #   E_i = Energy value of i"th channel

        order_0 = float(spectrum["header"]["Energy_Calibration_Offset"])
        order_1 = float(spectrum["header"]["Energy_Calibration_Gain"])
        order_2 = float(spectrum["header"]["Energy_Calibration_Quadratic"])
        order_3 = float(spectrum["header"]["Energy_Calibration_Cubic"])
        low_E = float(spectrum["header"]["Energy_Calibration_Low_Energy"])

        info = {
            "description": description,
            "timestamp": spectrum["header"]["Date-time_VAX"],
            "live_time": spectrum["header"]["Live_Time"],
            "real_time": spectrum["header"]["Total_time_per_real_time"],
            # The following commented out fields are PyRIID-only, which can't be stored in PCF.
            # They are shown here to explicitly communicate what would be lost in translation.
            #   - snr_target
            #   - snr_estimate
            #   - sigma
            #   - bg_counts
            #   - fg_counts
            #   - bg_counts_expected
            #   - fg_counts
            "total_counts": sum(spectrum["spectrum"]),
            "total_neutron_counts": spectrum["header"]["Total_Neutron_Counts"],
            "distance_cm": distance,
            "area_density": ad,
            "ecal_order_0": order_0,
            "ecal_order_1": order_1,
            "ecal_order_2": order_2,
            "ecal_order_3": order_3,
            "ecal_low_e": low_E,
            "atomic_number": an,
            "occupancy_flag": spectrum["header"]["Occupancy_Flag"],
            "tag": spectrum["header"]["Tag"],
        }

        infos.append(info)
        sources.append((category, title, source))
        spectra[i, :] = np.array(spectrum["spectrum"])

    n_sources = len(sources)
    n_samples = len(spectra)
    new_index = pd.MultiIndex.from_tuples(sources, names=SampleSet.SOURCES_MULTI_INDEX_NAMES)
    sources_df = pd.DataFrame(np.zeros((n_samples, n_sources)), columns=new_index)
    for i, key in enumerate(sources):
        sources_df[key].iloc[i] = 1.0

    ss = SampleSet()
    ss.spectra = pd.DataFrame(data=spectra)
    ss.sources = sources_df
    ss.info = pd.DataFrame(data=infos)
    ss.measured_or_synthetic = "synthetic",
    ss.detector_info["pcf_metadata"] = pcf_dict["header"]

    return ss


def _convert_header(header_dict: dict, header_def: dict):
    """Converts the header to a bytes object.

    Args:
        header_dict: Defines a dictionary of header values to be converted.
        header_def: Defines a dictionary of field and n_bytes values.

    Returns:
        A byte array containing the converted header_dict.

    Raises:
        None.
    """
    values = []
    for i, tar_len in zip(header_def["fields"], header_def["n_bytes"]):
        if "unused" not in i:
            value = header_dict[i]
            if isinstance(value, str):
                if len(value) < tar_len:
                    value = value.ljust(tar_len, " ")
                value = value.encode("utf-8")
            if i == "Tag":
                if value == b"":
                    value = b" "
            values.append(value)
        else:
            values.append(0.0)
    return struct.pack(header_def["mask"], *values)


def _spectrum_byte_offset(spectrum_index: int,
                          n_records_per_spectrum: int,
                          spec_rec_start_index: int = 83):
    """Gives byte offset in file for where spectrum should occur, where
    spectrum_index begins with index 1.

    Args:
        spectrum_index: Defines the index of the spectrum to be located.
        n_records_per_spectrum: Defines the number of records (channels) per spectrum.

    Returns:
        The integer byte offset of the desired spectrum.

    Raises:
        None.
    """
    return 256 * (spec_rec_start_index + n_records_per_spectrum * (spectrum_index - 1) - 1)


def _spectrum_to_bytes(spectrum: list):
    """Converts spectrum to bytes.

    Args:
        spectrum: Defines an array of floats.

    Returns:
        The bytes object representing the spectrum.

    Raises:
        None.
    """
    n_channels = len(spectrum)
    return struct.pack("{}f".format(n_channels), *spectrum)


def _dict_to_pcf(pcf_dict: dict, save_path: str, verbose: bool = True):
    """Converts dictionary of pcf information into pcf file.

    Args:
        pcf_dict: Defines a dictionary of pcf values.
        save_path: Defines the path at which to save the pcf.
        verbose: Determines whether or not to show verbose function output in terminal.

    Returns:
        The bytes object that is saved to file.

    Raises:
        None.
    """
    header = pcf_dict["header"]
    n_records_per_spectrum = header["NRPS"]
    n_spectra = len(pcf_dict["spectra"])

    file_bytes = _convert_header(header, HEADER_DEFINITIONS["DHS"])
    file_bytes += struct.pack("20s", b"DeviationPairsInFile")
    loc_first_spectra = _spectrum_byte_offset(1, n_records_per_spectrum)
    n_pad = 512 - len(file_bytes)
    file_bytes += bytes((" " * n_pad).encode("utf-8"))
    n_pad = loc_first_spectra - len(file_bytes)
    file_bytes += bytes(n_pad)

    # Add the spectra to file
    for i in range(n_spectra):
        spectrum_dict = pcf_dict["spectra"][i]
        spectrum_header_dict = spectrum_dict["header"]
        spectrum_header_bytes = _convert_header(spectrum_header_dict, SPECTRUM_DEFINITION)
        spectrum_bytes = _spectrum_to_bytes(spectrum_dict["spectrum"])
        file_bytes += spectrum_header_bytes + spectrum_bytes
    # save binary file
    with open(save_path, "wb") as fout:
        fout.write(file_bytes)


def _unpack_compressed_text_buffer(ctb, field_len=60):
    """Unpacks a compressed text buffer into title, description, and source."""
    if ord(ctb[0]) == 255:
        ctb_parts = ctb.split(ctb[0])
        title, description, source = ctb_parts[1:]
    else:
        title = ctb[:field_len]
        description = ctb[field_len:field_len*2]
        source = ctb[field_len*2:]
    title = title.strip()  # this is treated as the isotope
    # Note: description is described in PCF doc but currently does not appear accessible via API
    description = description.strip()
    source = source.strip()
    return title, description, source


def _pack_compressed_text_buffer(title, desc, source, field_len=60):
    """ Converts title, description, and source strings into a single, PCF-compatible array of 180
        characters to be put into the compressed text buffer.

        Each argument should by 60 characters or less.
        PyRIID does not use the delimiter approach for the compressed text buffer when converting
        from SMPL to PCF. Any part of the field exceeding 60 bytes will be truncated to fit.

        Args:
            title: the isotope of the record
            desc: custom user description
            source: the seed used to generate the sample, i.e., the inject source string
            field_len: the fixed length of each field if delimeters are not to be used

        Returns:
            A string of length 180.
    """
    assert field_len >= 3

    ctb_len = field_len * 3
    title = title.strip()
    title_len = len(title)
    desc = desc.strip()
    desc_len = len(desc)
    source = source.strip()
    source_len = len(source)

    if title_len <= field_len and desc_len <= field_len and source_len <= field_len:
        ctb = \
            f'{title:{field_len}.{field_len}}' + \
            f'{desc:{field_len}.{field_len}}' + \
            f'{source:{field_len}.{field_len}}'
    else:
        if title_len + desc_len + source_len <= ctb_len - 3:
            ctb = \
                '\xFF' + \
                f'{title:{title_len}.{title_len}}' + \
                '\xFF' + \
                f'{desc:{desc_len}.{desc_len}}' + \
                '\xFF' + \
                f'{source:{source_len}.{source_len}}'
        else:
            # Description is thrown away
            title_len = min(title_len, ctb_len - 3 - source_len)
            ctb = \
                '\xFF' + \
                f'{title:{title_len}.{title_len}}' + \
                '\xFF\xFF' + \
                f'{source:{source_len}.{source_len}}'

    if len(ctb) < ctb_len:
        ctb = ctb.ljust(ctb_len)

    return ctb


def _smpl_to_dict(ss: SampleSet):
    """Converts a Sampleset to a dictionary of values.

    Args:
        ss: Defines a Sampleset object to be converted.

    Returns:
        A dictionary containing the values from the Sampleset.

    Raises:
        None.
    """
    n_channels = ss.n_channels
    n_records_per_spectrum = int((n_channels / 64) + 1)

    if "pcf_metadata" in ss.detector_info and ss.detector_info["pcf_metadata"]:
        pcf_header = ss.detector_info["pcf_metadata"]
    else:
        pcf_header = {
            "NRPS": n_records_per_spectrum,
            "Version": "DHS",
            "last_mod_hash": " " * 7,
            "UUID": " " * 36,
            "Inspection": " " * 16,
            "Lane_Number": 0,
            "Measurement_Remark": " " * 26,
            "Intrument_Type": " " * 28,
            "Manufacturer": " " * 28,
            "Instrument_Model": " " * 18,
            "Instrument_ID": " " * 18,
            "Item_Description": " " * 20,
            "Item_Location": " " * 16,
            "Measurement_Coordinates": " " * 16,
            "Item_to_detector_distance": 0,
            "Occupancy_Number": 0,
            "Cargo_Type": " " * 16,
            "SRSI": 83,
            "DevType": "DeviationPairsInFile",
        }

    spectra = []

    isotopes, seeds = pd.Series(), pd.Series()
    if not ss.sources.empty:
        isotope_level_name = SampleSet.SOURCES_MULTI_INDEX_NAMES[1]
        seed_level_names = SampleSet.SOURCES_MULTI_INDEX_NAMES[2]
        if isotope_level_name in ss.sources.columns.names:
            isotopes = ss.get_labels(target_level=isotope_level_name)
        if seed_level_names in ss.sources.columns.names:
            seeds = ss.get_labels(target_level=seed_level_names)

    for i in range(ss.n_samples):
        title = isotopes[i] if not isotopes.empty else NO_ISOTOPE
        description = ss.info.description.fillna("").iloc[i]
        source = seeds[i] if not seeds.empty else NO_SEED
        compressed_text_buffer = _pack_compressed_text_buffer(title, description, source)

        header = {
            "Compressed_Text_Buffer": compressed_text_buffer,
            "Energy_Calibration_Low_Energy": ss.info.ecal_low_e.fillna(0).iloc[i],
            "Energy_Calibration_Offset": ss.info.ecal_order_0.fillna(0).iloc[i],
            "Energy_Calibration_Gain": ss.info.ecal_order_1.fillna(0).iloc[i],
            "Energy_Calibration_Quadratic": ss.info.ecal_order_2.fillna(0).iloc[i],
            "Energy_Calibration_Cubic": ss.info.ecal_order_3.fillna(0).iloc[i],
            "Live_Time": ss.info.live_time.fillna(0).iloc[i],
            "Total_time_per_real_time": ss.info.real_time.fillna(0).iloc[i],
            "Number_of_Channels": int(n_channels),
            "Date-time_VAX": ss.info.timestamp.fillna("").iloc[i],
            "Occupancy_Flag": ss.info.occupancy_flag.fillna(0).iloc[i],
            "Tag": ss.info.tag.fillna("").iloc[i],
            "Total_Neutron_Counts": ss.info.total_neutron_counts.fillna(0).iloc[i],
        }

        spectrum = ss.spectra.values[i, :]
        spectra.append({"header": header, "spectrum": spectrum})
    return {"header": pcf_header, "spectra": spectra}
