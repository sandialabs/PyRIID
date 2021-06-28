# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This module contains utilities for loading and saving SampleSet data files."""
import copy
import logging
import os
import pathlib
import pickle
import re
import warnings

import pandas as pd

from riid import DATA_PATH, DataDirectoryNotFoundError
from riid.sampleset import SampleSet


def _check_data_path():
    if not os.path.exists(DATA_PATH):
        raise DataDirectoryNotFoundError()


def check_iso_name(name: str):
    """ Validates whether or not the given string contains
        a properly formatted radioisotope name.

        Note that this function does NOT look up a string
        to determine if the string corresponds to a
        radioisotope that actually exists, it just checks
        the format.

        The regular expression used by this function
        looks for the following (in order):
            - 1 capital letter
            - 0 to 1 lowercase letters
            - 1 to 3 numbers
            - an optional "m" for metastable

        Examples of properly formatted isotope names:
          - Y88
          - Ba133
          - Ho166m

        Args:
          name: the string to be checked

        Returns:
          True if the string has a valid format,
          otherwise False.
    """
    validator = re.compile(r"^[A-Z]{1}[a-z]{0,1}[0-9]{1,3}m?$")
    other_valid_names = ["fiestaware"]
    match = validator.match(name)
    is_valid = match is not None or \
        name.lower() in other_valid_names
    return is_valid


def load_samples_from_file(file_path: str, verbose=1) -> SampleSet:
    """Load samples from the given file_path."""
    ss = None
    if os.path.isfile(file_path):
        message = "Found samples, loading '{}'.".format(file_path)
        if verbose:
            logging.info(message)
        try:
            raw_ss = read_hdf(file_path)
        except OSError:
            with open(file_path, "rb") as fin:
                raw_ss = pickle.load(fin)
        kwargs = raw_ss.__dict__
        # Make instance of most recent SampleSet using data from loaded file.
        ss = SampleSet(**kwargs)
    else:
        message = "No samples were found for the given parameters."
        if verbose:
            logging.info(message)
        ss = None
    return ss


def load_samples(detector: str, measured_or_synthetic: str, train_or_test: str, file_name: str,
                 verbose=1) -> SampleSet:
    """Load samples for a given detector, type, and file name.
    """
    file_path = os.path.join(DATA_PATH, detector, measured_or_synthetic, train_or_test, file_name)
    file_path = os.path.expanduser(file_path)
    ss = load_samples_from_file(file_path, verbose=verbose)
    return ss


def save_samples_to_file(ss: SampleSet, file_path: str, verbose=1):
    """Writes out the given sampleset to disk at the given path."""
    try:
        write_hdf(ss, file_path)
    except OSError:
        with open(file_path, "wb") as fout:
            pickle.dump(ss, fout)
    if verbose:
        logging.info(f"Saved SampleSet to '{file_path}'")


def save_samples(ss: SampleSet, file_name: str, detector: str = None,
                 measured_or_synthetic: str = None, purpose: str = None, verbose=1):
    """Save the given samples to the appropriate data directory location."""
    _check_data_path()

    if not ss:
        raise EmptySampleSetError("No samples were provided")

    if detector:
        ss.detector = detector
    if measured_or_synthetic:
        ss.measured_or_synthetic = measured_or_synthetic
    if purpose:
        ss.purpose = purpose

    output_dir = os.path.join(
        DATA_PATH,
        ss.detector,
        ss.measured_or_synthetic,
        ss.purpose
    )
    output_dir = os.path.expanduser(output_dir)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_path = os.path.join(
        DATA_PATH,
        ss.detector,
        ss.measured_or_synthetic,
        ss.purpose,
        file_name
    )
    save_samples_to_file(ss, output_path, verbose)


def save_detector_model(model_contents: str, model_hash: str, detector_name: str):
    """Save detector model
    """
    output_fn = os.path.join(
        DATA_PATH,
        detector_name,
        f"{model_hash}.dat"
    )
    with open(output_fn, "w") as fout:
        fout.write(model_contents)


def load_seeds(detector: str, measured_or_synthetic: str, file_name: str, verbose=1) -> SampleSet:
    """Load seeds for a given detector, type, and file_name.
    """
    _check_data_path()

    load_path = os.path.join(DATA_PATH, detector, measured_or_synthetic, "seed", file_name)
    load_path = os.path.abspath(os.path.expanduser(load_path))

    if not os.path.isfile(load_path):
        message = "No seeds were found for the given configuration at path {}".format(load_path)
        raise NoSeedsFoundError(message)

    if verbose:
        message = "Found seeds, loading '{}'.".format(load_path)
        logging.info(message)

    try:
        raw_seeds = read_hdf(load_path)
    except OSError:
        raw_seeds = pickle.load(open(load_path, "rb"))

    ss = SampleSet(**raw_seeds.__dict__)
    return ss


def save_seeds(ss: SampleSet, file_name: str, detector: str = None,
               measured_or_synthetic: str = None, verbose=1):
    """Load seeds for a given detector, type, and file_name."""
    _check_data_path()

    if not ss:
        raise EmptySampleSetError("No seeds were provided.")

    if detector:
        ss.detector = detector
    if measured_or_synthetic:
        ss.measured_or_synthetic = measured_or_synthetic

    save_samples(ss, file_name, verbose=verbose)


def read_hdf(file_name: str) -> SampleSet:
    """ Reads sampleset class from hdf binary format."""
    spectra = pd.read_hdf(file_name, "spectra")
    collection_information = pd.read_hdf(file_name, "collection_information")
    sources = pd.read_hdf(file_name, "sources")
    sources.columns = [i.split("___")[0] for i in sources.columns]
    features = pd.read_hdf(file_name, "features")
    prediction_probas = pd.read_hdf(file_name, "prediction_probas")
    other_info = pd.read_hdf(file_name, "other_info")
    kwargs = {
        "spectra": spectra,
        "collection_information": collection_information,
        "sources": sources,
        "features": features,
        "prediction_probas": prediction_probas,
    }
    for col in other_info.columns:
        kwargs.update({col: other_info.loc[0, col]})

    return SampleSet(**kwargs)


def write_hdf(ss: SampleSet, output_path: str):
    """ Writes sampleset class to hdf binary format."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ss.spectra.to_hdf(output_path, "spectra")
        ss.collection_information.to_hdf(output_path, "collection_information")

        sources = copy.copy(ss.sources)
        s = sources.columns.to_series()
        sources.columns = s + "____" + s.groupby(s).cumcount().astype(str).replace({'0': ''})
        sources.to_hdf(output_path, "sources")
        pd.DataFrame(ss.features).to_hdf(output_path, "features")
        ss.prediction_probas.to_hdf(output_path, "prediction_probas")

        # Get desired properties
        properties = [
            "detector_hash",
            "neutron_detector_hash",
        ]
        other_info = {
            "detector": ss.detector,
            "config": ss.config,
            "sensor_information": ss.sensor_information,
            "measured_or_synthetic": ss.measured_or_synthetic,
            "subtract_background": ss.subtract_background,
            "purpose": ss.purpose,
            "comments": ss.comments,
            "predictions": ss.predictions,
            "energy_bin_centers": ss.energy_bin_centers,
            "energy_bin_edges": ss._energy_bin_edges,
        }
        for prop in properties:
            if prop in ss.__dict__:
                other_info.update({prop: ss.__dict__[prop]})
            else:
                print(f"Not actually considering {prop}")

        df_other_info = pd.DataFrame(data=[other_info])

        df_other_info.to_hdf(output_path, "other_info")


class NoSeedsFoundError(Exception):
    pass


class EmptySampleSetError(Exception):
    pass
