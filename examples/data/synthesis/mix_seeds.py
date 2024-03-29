# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to generate synthetic gamma spectra from seeds."""
from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer

fg_seeds_ss, bg_seeds_ss = get_dummy_seeds().split_fg_and_bg()

mixed_fg_seeds_ss = SeedMixer(fg_seeds_ss, mixture_size=2)\
    .generate(n_samples=10)
mixed_bg_seeds_ss = SeedMixer(bg_seeds_ss, mixture_size=3)\
    .generate(n_samples=10)

print(mixed_fg_seeds_ss)
print(mixed_bg_seeds_ss)
