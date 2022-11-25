# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates energy calibration configuration for a SampleSet."""
import matplotlib.pyplot as plt
import numpy as np
from riid.data.labeling import BACKGROUND_LABEL
from riid.data.synthetic.static import StaticSynthesizer, get_dummy_sampleset

seeds_ss = get_dummy_sampleset(as_seeds=True)
static_syn = StaticSynthesizer()
# Separate foreground seeds from background seeds
seeds_labels = seeds_ss.get_labels()
fg_seeds_ss = seeds_ss[seeds_labels != BACKGROUND_LABEL]
bg_seeds_ss = seeds_ss[seeds_labels == BACKGROUND_LABEL]
_, _, ss = gss.generate(fg_seeds_ss, bg_seeds_ss)
ss.ecal_low_e = 0
ss.ecal_order_0 = 0
ss.ecal_order_1 = 3100
ss.ecal_order_2 = 0
ss.ecal_order_3 = 0

index = ss[ss.get_labels() == "Cs137"]
index = np.sum(index.spectra, axis=0).values.astype(int).argmax()
plt.subplot(2, 1, 1)
plt.step(
    np.arange(ss.n_channels),
    ss.spectra.values[index, :],
    where="mid"
)
plt.yscale("log")
plt.xlim(0, ss.n_channels)
plt.title(ss.get_labels()[index])
plt.xlabel("Channels")

energy_bins = np.linspace(0, 3100, 512)
channel_energies = ss.get_channel_energies(index=0, fractional_energy_bins=energy_bins)

plt.subplot(2, 1, 2)

plt.step(channel_energies, ss.spectra.values[index, :], where="mid")
plt.fill_between(
    energy_bins,
    0,
    ss.spectra.values[index, :],
    alpha=0.3,
    step="mid"
)

plt.yscale("log")
plt.title(ss.get_labels()[index])
plt.xlabel("Energy (keV)")
plt.vlines(661, 1e-6, 1e8)
plt.ylim(bottom=0.8, top=ss.spectra.values[index, :].max()*1.5)
plt.xlim(0, energy_bins[-1])
plt.subplots_adjust(hspace=1)
plt.show()
