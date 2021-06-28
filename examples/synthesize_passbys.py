# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""Example of generating synthetic passby gamma spectra from seeds."""
from time import time

import matplotlib.pyplot as plt
import numpy as np
from riid.data import load_seeds
from riid.synthetic import PassbySynthesizer

seeds = load_seeds("gm_561", "measured", "seven_isotopes.smpl")
pbs = PassbySynthesizer(seeds, fwhm_function_args=[.2, .5], events_per_seed=3)

tstart = time()
events = pbs.generate(verbose=1)
delay = time() - tstart

n_events = len(events)
print("Generated {} events in {:.1f} seconds ({} passbys/sec)".format(
    n_events,
    delay,
    n_events / delay)
)

ss = events[0]
ss.concat(events[1:])
plt.imshow(np.log10(ss.spectra.values[:, :].clip(1e-2)), aspect="auto")
plt.ylabel("Time")
plt.xlabel("Channel")
plt.show()

offset = 0
breaks = []
for iso in ss.source_types:
    print(f"Printing signal strength history of {iso}.")
    sub = ss.get_indices(ss.labels == iso)
    plt.plot(pbs.sample_interval * (np.arange(sub.n_samples) + offset), sub.sigma, label=f"{iso}", lw=1)
    offset += sub.n_samples
    breaks.append(offset*pbs.sample_interval)

plt.vlines(breaks, 0, 1e4, linestyle="--", color="grey")


plt.yscale("log")
plt.ylim([.1, 1e4])
plt.xlim(0, breaks[-1])
plt.ylabel("Sigma")
plt.xlabel("Time")
plt.legend(frameon=False)
plt.show()

