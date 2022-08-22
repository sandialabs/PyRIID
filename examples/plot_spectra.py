# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to plot gamma spectra."""
from riid.data.synthetic.static import get_dummy_sampleset
from riid.visualize import plot_spectra

seeds_ss = get_dummy_sampleset(as_seeds=True)

plot_spectra(seeds_ss, ylim=(None, None), in_energy=True)
