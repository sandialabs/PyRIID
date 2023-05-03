# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to generate synthetic gamma spectra from seeds."""
from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer

SYNTHETIC_DATA_CONFIG = {
    "samples_per_seed": 10000,
    "bg_cps": 10,
    "snr_function": "uniform",
    "snr_function_args": (1, 100),
    "live_time_function": "uniform",
    "live_time_function_args": (0.25, 10),
    "apply_poisson_noise": True,
    "return_fg": True,
    "return_bg": True,
    "return_gross": True,
}
fg_seeds_ss, bg_seeds_ss = get_dummy_seeds().split_fg_and_bg()

mixed_bg_seed_ss = SeedMixer(bg_seeds_ss, mixture_size=3)\
    .generate(1)

static_synth = StaticSynthesizer(**SYNTHETIC_DATA_CONFIG)
fg_ss, bg_ss, gross_ss = static_synth.generate(fg_seeds_ss, mixed_bg_seed_ss)
""" |      |         |
    |      |         |> gross samples
    |      |> background-only samples
    |> source-only samples
"""
print(fg_ss)
print(bg_ss)
print(gross_ss)
