# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Example of generating synthetic passby gamma spectra from seeds."""
import sys

import matplotlib.pyplot as plt
import numpy as np

from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.passby import PassbySynthesizer

if len(sys.argv) == 2:
    import matplotlib
    matplotlib.use("Agg")


fg_seeds_ss, bg_seeds_ss = get_dummy_seeds().split_fg_and_bg()
pbs = PassbySynthesizer(
    sample_interval=0.5,
    fwhm_function_args=(5, 5),
    snr_function_args=(100, 100),
    dwell_time_function_args=(5, 5),
    events_per_seed=1,
    return_fg=False,
    return_gross=True,
)

events = pbs.generate(fg_seeds_ss, bg_seeds_ss)
_, gross_passbys = list(zip(*events))
passby_ss = gross_passbys[0]
passby_ss.concat(gross_passbys[1:])

passby_ss.sources.drop(bg_seeds_ss.sources.columns, axis=1, inplace=True)
passby_ss.normalize_sources()
passby_ss.normalize(p=2)

plt.imshow(passby_ss.spectra.values, aspect="auto")
plt.ylabel("Time")
plt.xlabel("Channel")
plt.show()

offset = 0
breaks = []
labels = passby_ss.get_labels()
distinct_labels = set(labels)
# Iterate over set of unique labels
for iso in distinct_labels:
    sub = passby_ss[labels == iso]
    plt.plot(
        pbs.sample_interval * (np.arange(sub.n_samples) + offset),
        sub.info.snr,
        label=f"{iso}",
        lw=1
    )
    offset += sub.n_samples
    breaks.append(offset*pbs.sample_interval)

plt.vlines(breaks, 0, 1e4, linestyle="--", color="grey")

plt.yscale("log")
plt.ylim([.1, 1e4])
plt.xlim(0, breaks[-1])
plt.ylabel("SNR")
plt.xlabel("Time")
plt.legend(frameon=False)
plt.show()
