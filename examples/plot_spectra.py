# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This example demonstrates how to plot gamma spectra."""
from riid.data import load_seeds
from riid.visualize import plot_spectra

seeds = load_seeds("gm_561", "measured", "seven_isotopes.smpl")
plot_spectra(seeds, ylim=(1e-5, 1), is_in_energy=False)

# ecal required to plot as energy
seeds.ecal_order_0 = 0
seeds.ecal_order_1 = 3000
seeds.ecal_order_2= 0
seeds.ecal_order_3 = 0
seeds.ecal_low_e = 0

plot_spectra(seeds, ylim=(1e-5, 1), is_in_energy=True)
