# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module just aliases the `sampleset` module."""
import numpy as np


def get_expected_spectra(seeds: np.ndarray, expected_counts: np.ndarray) -> np.ndarray:
    """Multiply a 1-D array of expected counts by either a 1-D array or 2-D
    matrix of seed spectra.

    The dimension(s) of the seed array(s), `seeds`, is expanded to be `(m, n, 1)` where:

    - m = # of seeds
    - n = # of channels

    and the final dimension is added in order to facilitate proper broadcasting
    The dimension of the `expected_counts` must be 1, but the length `p` can be
    any positive number.

    The resulting expected spectra will be of shape `(m x p, n)`.
    This representings the same number of channels `n`, but each expected count
    value, of which there were `p`, will be me multiplied through each seed spectrum,
    of which there were `m`.
    All expected spectra matrices for each seed are then concatenated together
    (stacked), eliminating the 3rd dimension.
    """
    if expected_counts.ndim != 1:
        raise ValueError("Expected counts array must be 1-D.")
    if expected_counts.shape[0] == 0:
        raise ValueError("Expected counts array cannot be empty.")
    if seeds.ndim > 2:
        raise InvalidSeedError("Seeds array must be 1-D or 2-D.")

    expected_spectra = np.concatenate(
        seeds * expected_counts[:, np.newaxis, np.newaxis]
    )

    return expected_spectra


class InvalidSeedError(Exception):
    """Seed spectra data structure is not 1- or 2-dimensional."""
    pass
