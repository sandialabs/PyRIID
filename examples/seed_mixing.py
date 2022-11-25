# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to generate synthetic gamma spectra from seeds."""
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import get_dummy_sampleset

seeds_ss = get_dummy_sampleset(as_seeds=True)
seeds_ss.normalize()

mixer = SeedMixer(
    mixture_size=2,
    min_source_contribution=0.1,
)
two_mix_seeds_ss = mixer.generate(seeds_ss, n_samples=5)

print(two_mix_seeds_ss)
