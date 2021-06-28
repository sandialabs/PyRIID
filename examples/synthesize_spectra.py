# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This example demonstrates how to generate synthetic gamma spectra from seeds."""
from time import time

from riid.data import load_seeds
from riid.synthetic import GammaSpectraSynthesizer

seeds = load_seeds("gm_561", "measured", "seven_isotopes.smpl")
gss = GammaSpectraSynthesizer(seeds)

tstart = time()
ss = gss.generate(verbose=1)
delay = time() - tstart

n_samples = ss.n_samples
summary = "Generated {} samples in {:.2f}s ({:.2f} samples/sec).".format(
    n_samples,
    delay,
    n_samples / delay
)
print(summary)
print(gss)
