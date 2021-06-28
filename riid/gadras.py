# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This module contains utilities for working with GADRAS and related files."""
import copy
import re
import struct
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from riid.sampleset import SampleSet


# Define PCF definition per PCF ICD
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
    }
)

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


def pcf_to_smpl(pcf_file_path, verbose=False):
    """ Converts pcf file to sampleset object.

    Inputs:
        pcf_file_path: path to pcf file

    Returns:
        sampleset: object containing information of pcf file in sampleset format
    """
    return _pcf_dict_to_sampleset(_pcf_to_dict(pcf_file_path, verbose))


def smpl_to_pcf(ss: SampleSet, save_path, verbose=True):
    """ Saves a sampleset to a pcf file
    """
    return _dict_to_pcf(_smpl_to_dict(ss), save_path, verbose=verbose)


def _get_srsi(file_bytes):
    """ Obtains the PCF file Spectral Records Start Index (SRSI).

    Inputs:
        file_bytes: byte array of file contents

    Returns:
        return_value: String indicator for what Deviation pairs type is being used.
        srsi: Spectral Records Start Index.  Index in file where spectra records begin
        after skipping past deviation pairs.
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
    return(return_value, srsi)


def _get_spectrum_header_offset(spectrum_number, srsi, nrps):
    """ Calculates the PCF header offset for spectrum.

    Inputs:
        spectrum_number: number of spectrum to obtain offset of header for.
        srsi: Spectral record start index.
        nrps: number of records per spectrum

    Returns:
        offset of spectrum header
    """
    return 256*(srsi + nrps * (spectrum_number - 1) - 1)


def _read_header(header_bytes, header_def):
    """ Converts bytes of header using header definition.

    Inputs:
        header_bytes: byte array of values of header.
        header_def: Dictionary defining the: mask, field names, and lengths of each entry.

    Returns:
        header: dictionary containing the content of the header.
    """
    header_values = struct.unpack(header_def["mask"], header_bytes[:sum(header_def["n_bytes"])])
    header = {}
    for field, value in zip(header_def["fields"], header_values):
        if isinstance(value, bytes) and field != "last_mod_hash":
            value = value.decode("utf-8", "ignore")
        header.update({field: value})
    return header


def _read_spectra(data, n_rec_per_spec, spec_rec_start_indx, verbose=False):
    """ Reads spectra from PCF file.

    Inputs:
        data: byte array of pcf file contents.
        n_rec_per_spec: Defines the number of records per spectrum. (NRPS in pcf ICD)
        spec_rec_start_index: Defines spectral record start index. (SRSI in pcf ICD)
        verbose: Boolean flag which will enable or disable status updates of reading in spectra.

    Returns:
        spectra: list of dictionaries containing pcf file spectra information.
    """
    num_samples = int((len(data) - 256 * (spec_rec_start_indx - 1)) / (256 * n_rec_per_spec))
    spectra = []
    # The range below is due to the definition of the spectrum offset in
    # pcf definition document assuming a first index of 1
    progress_iterable = tqdm(
        range(1, num_samples + 1),
        leave=False,
        desc="Spectrum",
        disable=not(verbose)
    )
    for spectrum_number in progress_iterable:
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

        spectrum = {"header": spectrum_header, "spectrum": np.array(values)}
        spectra.append(spectrum)

    return spectra


def _pcf_to_dict(pcf_file_path, verbose=False):
    """ Converts .pcf file into a python dictionary

    returns a dict with keys "header" and "spectra"

    Header provides metadata about the collection.

    spectra is a list of "header" and "spectrum" pairs for each spectrum in file.
    """
    pcf_data = np.fromfile(pcf_file_path, dtype=np.uint8)

    version = struct.unpack("3s", pcf_data[2:5])[0].decode("utf-8")
    header_def = HEADER_DEFINITIONS[version]
    deviation_values = struct.unpack("5120f", pcf_data[512:512+20480])
    if not set(deviation_values) == {0} and verbose:
        print("Devitaion values exist in file")
    header = _read_header(pcf_data[:256], header_def)
    dev_type, spec_rec_start_indx = _get_srsi(pcf_data)
    header.update({"SRSI": spec_rec_start_indx, "DevType": dev_type})
    spectra = _read_spectra(pcf_data, header["NRPS"], spec_rec_start_indx, verbose=verbose)

    return {"header": header, "spectra": spectra}


def _pcf_dict_to_sampleset(pcf_dict):
    """ Converts pcf dictionary into a SampleSet
    """

    if pcf_dict["spectra"]:
        num_spectra = len(pcf_dict["spectra"])
        num_channels = len(pcf_dict["spectra"][0]["spectrum"])

        sources = []
        collection_information = []

        spectra = np.ndarray((num_spectra, num_channels))

        for i in range(0, num_spectra):
            foreground = pcf_dict["spectra"][i]
            f_text_buffer = foreground["header"]["Compressed_Text_Buffer"]
            fg_counts = sum(foreground["spectrum"])

            total_counts = sum(foreground["spectrum"])
            live_time = foreground["header"]["Live_Time"]

            distance_search = re.search("@ ([0-9,.]+)", f_text_buffer)
            if distance_search:
                distance = float(distance_search.group(1)) / 100
            else:
                distance = np.nan

            source_string_initial = f_text_buffer.split(",")[0]
            source_string_initial = source_string_initial.split("{")[0]

            an_finds = re.findall("an=([0-9]+)", f_text_buffer)
            if an_finds:
                an = int(an_finds[0])
            else:
                an = None
            ad_finds = re.findall("ad=([0-9, .]+)", f_text_buffer)
            if ad_finds:
                ad_string = ad_finds[0].replace(",", "")
                ad = float(ad_string)
            else:
                ad = None

            # if source_string_initial:
            #     if f_text_buffer[:4] == "1kgU":
            #         source_string_initial = f_text_buffer[4:]
            #     source = _format_isotope_name(source_string_initial)
            source = f_text_buffer.strip()
            if source == "":
                source = "background"

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

            order_0 = float(foreground["header"]["Energy_Calibration_Offset"])
            order_1 = float(foreground["header"]["Energy_Calibration_Gain"])
            order_2 = float(foreground["header"]["Energy_Calibration_Quadratic"])
            order_3 = float(foreground["header"]["Energy_Calibration_Cubic"])
            low_E = float(foreground["header"]["Energy_Calibration_Low_Energy"])

            info = {
                "live_time": live_time,
                "total_counts": total_counts,
                "fg_counts": fg_counts,
                "distance": distance,
                "atomic_number": an,
                "area_density": ad,
                "ecal_order_0": order_0,
                "ecal_order_1": order_1,
                "ecal_order_2": order_2,
                "ecal_order_3": order_3,
                "ecal_low_e": low_E,
                "date-time": foreground["header"]["Date-time_VAX"],
                "real_time": foreground["header"]["Total_time_per_real_time"],
                "occupancy_flag": foreground["header"]["Occupancy_Flag"],
                "tag": foreground["header"]["Tag"],
                "total_neutron_counts": foreground["header"]["Total_Neutron_Counts"],
                "descr": f_text_buffer,
                "count_rate": total_counts / live_time
            }

            collection_information.append(info)
            sources.append(source)
            spectra[i, :] = np.array(foreground["spectrum"])

        sources_df = pd.DataFrame()
        for source in set(sources):
            sources_df[source] = np.isin(sources, source)
        sources_df = sources_df[sources]
        sources_df["label"] = sources

        sample_set = SampleSet(
            spectra=pd.DataFrame(data=spectra),
            sources=sources_df,
            collection_information=pd.DataFrame(data=collection_information),
            measured_or_synthetic="synthetic",
            _pcf_metadata=pcf_dict["header"]
        )
        sample_set.energy_bin_centers = sample_set._get_energy_centers()
        # reorder sources table columns to match the order the seeds were read in

        return sample_set


def _format_isotope_name(name):
    """ places isotope in format of alpha followed by numeric (eg. Pu239 instead of 239Pu)
    """

    alpha_part = re.search("[a-z]+", name.lower())
    numeric_part = re.search("[0-9]+", name)

    if alpha_part and numeric_part:
        name = alpha_part.group(0).capitalize() + numeric_part.group(0)
    elif alpha_part:
        name = alpha_part.group(0).capitalize()
    elif numeric_part:
        name = numeric_part.group(0)
    else:
        name = "Unknown"
    return name


def _convert_header(header_dict, header_def):
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


def _spectrum_byte_offset(spectrum_index, n_records_per_spectrum, spec_rec_start_index=83):
    """ Gives byte offset in file for where spectrum should occur
    spectrum_index begins with index 1

    """
    return 256 * (spec_rec_start_index + n_records_per_spectrum * (spectrum_index - 1) - 1)


def _spectrum_to_bytes(spectrum):
    """ Converts spectrum to bytes
    """
    n_channels = len(spectrum)
    return struct.pack("{}f".format(n_channels), *spectrum)


def _dict_to_pcf(pcf_dict, save_path, verbose=True):
    """  Converts dictionary of pcf information into pcf file
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
    return file_bytes


def _smpl_to_dict(ss: SampleSet):
    n_samples = ss.n_samples
    n_channels = ss.n_channels
    n_records_per_spectrum = int((n_channels / 64) + 1)

    if ss._pcf_metadata:
        pcf_header = ss._pcf_metadata
    else:
        pcf_header = {
            "Cargo_Type": "                ",
            "DevType": "DeviationPairsInFile",
            "Inspection": "                ",
            "Instrument_ID": "                  ",
            "Instrument_Model": "                  ",
            "Intrument_Type": "                            ",
            "Item_Description": "                    ",
            "Item_Location": "                ",
            "Item_to_detector_distance": 0,
            "Lane_Number": 0,
            "Manufacturer": "                            ",
            "Measurement_Coordinates": "                ",
            "Measurement_Remark": "                          ",
            "NRPS": n_records_per_spectrum,
            "Occupancy_Number": 0,
            "SRSI": 83,
            "UUID": "                                    ",
            "Version": "DHS",
            "last_mod_hash": b"       "
        }

    spectra = []

    for i in range(n_samples):
        row, label = (ss.collection_information["descr"].fillna("").values[i], ss.labels[i])
        if row:
            descr = row
        else:
            descr = label

        header = {
            "Compressed_Text_Buffer": descr,
            "Energy_Calibration_Low_Energy": ss.ecal_low_e.fillna(0).iloc[i],
            "Energy_Calibration_Offset": ss.ecal_order_0.fillna(0).iloc[i],
            "Energy_Calibration_Gain": ss.ecal_order_1.fillna(0).iloc[i],
            "Energy_Calibration_Quadratic": ss.ecal_order_2.fillna(0).iloc[i],
            "Energy_Calibration_Cubic": ss.ecal_order_3.fillna(0).iloc[i],
            "Live_Time": ss.live_time.fillna(0).iloc[i],
            "Total_time_per_real_time": ss.collection_information["real_time"].fillna(0).iloc[i],
            "Number_of_Channels": int(n_channels),
            "Date-time_VAX": ss.collection_information["date-time"].fillna("").iloc[i],
            "Occupancy_Flag": ss.collection_information.occupancy_flag.fillna(0).iloc[i],
            "Tag": ss.collection_information.tag.fillna("").iloc[i],
            "Total_Neutron_Counts":
                ss.collection_information.total_neutron_counts.fillna(0).iloc[i],
        }

        spectrum = ss.spectra.values[i, :]
        spectra.append({"header": header, "spectrum": spectrum})
    return {"header": pcf_header, "spectra": spectra}


def _run_gadras_string_inject(path_to_detector, path_to_setup, output_path, verbose=True):
    PATH_TO_INJECT_EXE = "C:\\GADRAS\\Program\\InjectSource.exe"
    inputs = [PATH_TO_INJECT_EXE, path_to_detector, path_to_setup, output_path]

    output = subprocess.getoutput(inputs)
    if verbose:
        print(output)


def make_seeds(source_strings, detector_dir, detector_name, labels=None, tag=None):
    seed_pcf_path = tempfile.TemporaryFile()
    seed_pcf_path.name = "seeds.pcf"
    seed_list_path = tempfile.TemporaryFile()
    seed_list_path.name = "seeds_list.csv"

    column_titles = ["source", "real_time", "poisson", "distance", "height", "include_bg"]
    seeds = []
    for source_string in source_strings:
        seeds.append(
            {
                "source": source_string,
                "real_time": 1,
                "poisson": False,
                "distance": 5000,
                "height": 56,
                "include_bg": False,
            }
        )
    df = pd.DataFrame(seeds)
    df[column_titles].to_csv(seed_list_path.name, sep="\t", index=False, header=column_titles)

    # Generate seeds.pcf file
    _run_gadras_string_inject(detector_dir, seed_list_path.name, seed_pcf_path.name, True)

    seeds = pcf_to_smpl(seed_pcf_path, True)

    if labels:
        target_labels = copy.deepcopy(labels)
        seeds.labels = target_labels
        target_labels.append("label")
        seeds._sources.columns = target_labels

    seeds.purpose = "seed"
    seeds.measured_or_synthetic = "synthetic"
    seeds.to_pmf()

    return seeds
