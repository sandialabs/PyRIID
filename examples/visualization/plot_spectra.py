# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to plot gamma spectra."""
import sys

from riid.data.synthetic import get_dummy_seeds
from riid.visualize import plot_spectra

if len(sys.argv) == 2:
    import matplotlib
    matplotlib.use("Agg")

seeds_ss = get_dummy_seeds()

plot_spectra(seeds_ss, ylim=(None, None), in_energy=True)
