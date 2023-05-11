# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to compare sample sets."""
import sys

from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer
from riid.visualize import plot_ss_comparison

if len(sys.argv) == 2:
    import matplotlib
    matplotlib.use("Agg")

SYNTHETIC_DATA_CONFIG = {
    "samples_per_seed": 100,
    "bg_cps": 100,
    "snr_function": "uniform",
    "snr_function_args": (1, 100),
    "live_time_function": "uniform",
    "live_time_function_args": (0.25, 10),
    "apply_poisson_noise": True,
    "return_fg": False,
    "return_gross": True,
}
fg_seeds_ss, bg_seeds_ss = get_dummy_seeds()\
    .split_fg_and_bg()
mixed_bg_seed_ss = SeedMixer(bg_seeds_ss, mixture_size=3)\
    .generate(1)

_, _, gross_ss1 = StaticSynthesizer(**SYNTHETIC_DATA_CONFIG)\
    .generate(fg_seeds_ss, mixed_bg_seed_ss)
_, _, gross_ss2 = StaticSynthesizer(**SYNTHETIC_DATA_CONFIG)\
    .generate(fg_seeds_ss, mixed_bg_seed_ss)

ss1_stats, ss2_stats, col_comparisons = gross_ss1.compare_to(gross_ss2,
                                                             density=False)
plot_ss_comparison(ss1_stats,
                   ss2_stats,
                   col_comparisons,
                   "live_time",
                   show=True)

ss1_stats, ss2_stats, col_comparisons = gross_ss1.compare_to(gross_ss2,
                                                             density=True)
plot_ss_comparison(ss1_stats,
                   ss2_stats,
                   col_comparisons,
                   "total_counts",
                   show=True)
