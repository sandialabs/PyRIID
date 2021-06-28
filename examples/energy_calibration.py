# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This example demonstrates energy calibration configuration for a SampleSet."""
import matplotlib.pyplot as plt
import numpy as np
from riid.data import load_seeds
from riid.synthetic import GammaSpectraSynthesizer

seeds = load_seeds("gm_561", "measured", "seven_isotopes.smpl", False)
gss = GammaSpectraSynthesizer(seeds)
ss = gss.generate(verbose=1)
ss.ecal_low_e = 0
ss.ecal_order_0 = 0
ss.ecal_order_1 = 3100
ss.ecal_order_2 = 0
ss.ecal_order_3 = 0

index = ss.get_indices((ss.labels == "Cs137")).total_counts.astype(int).idxmax()
plt.subplot(2, 1, 1)
plt.step(
    np.arange(ss.n_channels),
    ss.spectra.values[index, :],
    where="mid"
)
plt.yscale("log")
plt.xlim(0, ss.n_channels)
plt.title(ss.labels[index])
plt.xlabel("Channels")

energy_bins = np.linspace(0, 3100, 512)
ss.to_energy(energy_bins)

plt.subplot(2, 1, 2)

plt.step(energy_bins, ss.spectra.values[index, :], where="mid")
plt.fill_between(
    energy_bins,
    0,
    ss.spectra.values[index, :],
    alpha=0.3,
    step="mid"
)

plt.yscale("log")
plt.title(ss.labels[index])
plt.xlabel("Energy (keV)")
plt.vlines(661, 1e-6, 1e8)
plt.ylim(bottom=0.8, top=ss.spectra.values[index, :].max()*1.5)
plt.xlim(0, energy_bins[-1])
plt.subplots_adjust(hspace=1)
plt.show()
