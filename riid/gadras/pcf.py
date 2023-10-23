# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains utilities for working with GADRAS PCF files."""
import struct
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import tqdm

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


def _get_srsi(file_bytes: list) -> Tuple[str, int]:
    """Obtains the PCF file Spectral Records Start Index (SRSI).

    Args:
        file_bytes: byte array of the pcf file contents

    Returns:
        Tuple containing:

        - return_value: string indicator for what Deviation pairs type is being used
        - srsi: Spectral Records Start Index, the index where spectra records begin
            after skipping past deviation pairs
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


def _get_spectrum_header_offset(spectrum_number: int, srsi: int, nrps: int) -> int:
    """Calculate the PCF header offset for a spectrum.

    Args:
        spectrum_number: number of spectrum for which to the
            obtain offset of header
        srsi: spectral record start index
        nrps: number of records (channels) per spectrum

    Returns:
        Integer representing the offset of the spectrum header
    """
    return 256 * (srsi + nrps * (spectrum_number - 1) - 1)


def _read_header(header_bytes: list, header_def: dict):
    """Converts bytes of header using header definition.

    Args:
        header_bytes: byte array of the values of the header
        header_def: dictionary defining the mask, field names, and lengths of each entry

    Returns:
        Dictionary containing the content of the header
    """
    header_values = struct.unpack(
        header_def["mask"],
        header_bytes[:sum(header_def["n_bytes"])]
    )
    header = {}
    for field, value in zip(header_def["fields"], header_values):
        if isinstance(value, bytes) and field != "last_mod_hash":
            value = value.decode("utf-8", "ignore")
        elif field == "last_mod_hash":
            value = str(value)
        header.update({field: value})
    return header


def _read_spectra(data: list, n_rec_per_spec: int, spec_rec_start_indx: int) -> List[dict]:
    """Read spectra from a PCF.

    Args:
        data: byte array of pcf file contents
        n_rec_per_spec: number of records per spectrum (NRPS in PCF ICD)
        spec_rec_start_index: spectral record start index (SRSI in PCF ICD)

    Returns:
        List of dictionaries containing pcf file spectra information
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


def _pcf_to_dict(pcf_file_path: str, verbose: bool = False) -> dict:
    """Convert a PCF into a dictionary representation.

    Args:
        pcf_file_path: path to the PCF to be converted
        verbose: whether to show detailed output

    Returns:
        Dictionary with keys "header" and "spectra"
    """
    pcf_data = np.fromfile(pcf_file_path, dtype=np.uint8)
    version = struct.unpack("3s", pcf_data[2:5])[0].decode("utf-8")
    header_def = HEADER_DEFINITIONS[version]
    has_deviation_pairs = "DeviationPairsInFile" in header_def or \
        "DeviationPairsInFileCompressed" in header_def
    if has_deviation_pairs:
        deviation_values = struct.unpack("5120f", pcf_data[512:512+20480])
        if not set(deviation_values) == {0} and verbose:
            print("Deviation pairs exist in file")
    # Header provides metadata about the collection.
    # Spectra is a list of "header" and "spectrum" pairs for each spectrum in file.
    header = _read_header(pcf_data[:256], header_def)
    dev_type, spec_rec_start_indx = _get_srsi(pcf_data)
    header.update({"SRSI": spec_rec_start_indx, "DevType": dev_type})
    spectra = _read_spectra(pcf_data, header["NRPS"], spec_rec_start_indx)

    return {"header": header, "spectra": spectra}


def _convert_header(header_dict: dict, header_def: dict) -> bytes:
    """Convert the header to a bytes object.

    Args:
        header_dict: dictionary of header values to be converted
        header_def: dictionary of field and n_bytes values

    Returns:
        `bytes` object containing the converted header dictionary
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


def _spectrum_byte_offset(spectrum_index: int, n_records_per_spectrum: int,
                          spec_rec_start_index: int = 83) -> int:
    """Get the byte offset in the file where the spectrum should occur, where
    spectrum_index begins with index equals 1.

    Args:
        spectrum_index: index of the spectrum to be located
        n_records_per_spectrum: number of records (channels) per spectrum

    Returns:
        The integer byte offset of the desired spectrum
    """
    return 256 * (spec_rec_start_index + n_records_per_spectrum * (spectrum_index - 1) - 1)


def _dict_to_pcf(pcf_dict: dict, save_path: str, verbose=False):
    """Convert PCF dictionary representation to PCF binary.

    Args:
        pcf_dict: dictionary of PCF values
        save_path: path at which to save the PCF
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
    # save binary file
    with open(save_path, "wb") as fout:
        fout.write(file_bytes)

    # Add the spectra to file
    sample_range = range(n_spectra)
    if verbose:
        sample_range = tqdm.tqdm(
            sample_range,
            desc="Writing to file"
        )

    with open(save_path, "ab") as fout:
        for i in sample_range:
            spectrum_dict = pcf_dict["spectra"][i]
            spectrum_header_dict = spectrum_dict["header"]
            spectrum_header_bytes = _convert_header(spectrum_header_dict, SPECTRUM_DEFINITION)
            n_channels = len(spectrum_dict["spectrum"])
            spectrum_bytes = struct.pack("{}f".format(n_channels), *spectrum_dict["spectrum"])
            file_bytes = spectrum_header_bytes + spectrum_bytes
            fout.write(file_bytes)


def _unpack_compressed_text_buffer(ctb, field_len=60) -> Tuple[str, str, str]:
    """Unpack a compressed text buffer into title, description, and source."""
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


def _pack_compressed_text_buffer(title, desc, source, field_len=60) -> str:
    """Convert title, description, and source strings into a single, PCF-compatible array of 180
    characters to be put into the compressed text buffer.

    Each argument should by 60 characters or less.
    PyRIID does not use the delimiter approach for the compressed text buffer when converting
    from `SampleSet` to PCF. Any part of the field exceeding 60 bytes will be truncated to fit.

    Args:
        title: isotope of the record
        desc: custom user description
        source: seed used to generate the sample, i.e., the inject source string
        field_len: fixed length of each field if delimeters are not to be used

    Returns:
        String of length 180
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
            f"{title:{field_len}.{field_len}}" + \
            f"{desc:{field_len}.{field_len}}" + \
            f"{source:{field_len}.{field_len}}"
    else:
        if title_len + desc_len + source_len <= ctb_len - 3:
            ctb = \
                "\xFF" + \
                f"{title:{title_len}.{title_len}}" + \
                "\xFF" + \
                f"{desc:{desc_len}.{desc_len}}" + \
                "\xFF" + \
                f"{source:{source_len}.{source_len}}"
        else:
            # Description is thrown away
            title_len = min(title_len, ctb_len - 3 - source_len)
            ctb = \
                "\xFF" + \
                f"{title:{title_len}.{title_len}}" + \
                "\xFF\xFF" + \
                f"{source:{source_len}.{source_len}}"

    if len(ctb) < ctb_len:
        ctb = ctb.ljust(ctb_len)

    return ctb
