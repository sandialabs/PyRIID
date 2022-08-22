# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to generate synthetic gamma spectra from seeds."""
from riid.data.labeling import BACKGROUND_LABEL
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer, get_dummy_sampleset

SYNTHETIC_DATA_CONFIG = {
    "samples_per_seed": 10,
    "background_cps": 10,
    "snr_function": "uniform",
    "snr_function_args": (1, 100),
    "live_time_function": "uniform",
    "live_time_function_args": (0.25, 10),
    "apply_poisson_noise": True,
    "balance_level": "Isotope",
}
seeds_ss = get_dummy_sampleset(as_seeds=True)
seed_labels = seeds_ss.get_labels()
fg_seeds_ss = seeds_ss[seed_labels != BACKGROUND_LABEL]
bg_seeds_ss = seeds_ss[seed_labels == BACKGROUND_LABEL]
static_synth = StaticSynthesizer(**SYNTHETIC_DATA_CONFIG)

# Synthesize foreground-only samples
fg_ss, _, _ = static_synth.generate(fg_seeds_ss, bg_seeds_ss)

# Synthesize background-only samples
_, bg_ss, _ = static_synth.generate(fg_seeds_ss, bg_seeds_ss)

# Synthesize gross samples
_, _, gross_ss = static_synth.generate(fg_seeds_ss, bg_seeds_ss)

# Mix foregrounds
mixer = SeedMixer(
    mixture_size=2,
    min_source_contribution=0.1,
)
two_mix_seeds_ss = mixer.generate(seeds_ss, n_samples=100)
two_mix_ss = static_synth.generate(two_mix_seeds_ss, bg_seeds_ss)
