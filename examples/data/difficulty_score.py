# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to compute the difficulty of a given SampleSet."""
from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer

fg_seeds_ss, bg_seeds_ss = get_dummy_seeds().split_fg_and_bg()
mixed_bg_seed_ss = SeedMixer(bg_seeds_ss, mixture_size=3)\
    .generate(1)

static_synth = StaticSynthesizer(
    samples_per_seed=500,
    snr_function="uniform",
)
easy_ss, _ = static_synth.generate(fg_seeds_ss, mixed_bg_seed_ss)

static_synth.snr_function = "log10"
medium_ss, _ = static_synth.generate(fg_seeds_ss, mixed_bg_seed_ss)

static_synth.snr_function_args = (.00001, .1)
hard_ss, _ = static_synth.generate(fg_seeds_ss, mixed_bg_seed_ss)

easy_score = easy_ss.difficulty_score
print(f"Difficulty score for Uniform:           {easy_score:.5f}")
medium_score = medium_ss.difficulty_score
print(f"Difficulty score for Log10:             {medium_score:.5f}")
hard_score = hard_ss.difficulty_score
print(f"Difficulty score for Log10 Low Signal:  {hard_score:.5f}")
