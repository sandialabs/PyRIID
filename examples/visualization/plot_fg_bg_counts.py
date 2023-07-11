# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to plot gamma spectra."""
import sys

from riid.data.synthetic import get_dummy_seeds
from riid.visualize import plot_fg_and_bg_spectra
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer

if len(sys.argv) == 2:
    import matplotlib
    matplotlib.use("Agg")

fg_seeds_ss, bg_seeds_ss = get_dummy_seeds().split_fg_and_bg()
mixed_bg_seed_ss = SeedMixer(bg_seeds_ss, mixture_size=3).generate(10)

static_synth = StaticSynthesizer(
    samples_per_seed=100,
    snr_function="log10",
    snr_function_args=(50, 100),
    return_fg=True,
    return_gross=False,
    return_bg=True
)
fg_ss, bg_ss, _ = static_synth.generate(fg_seeds_ss, mixed_bg_seed_ss)

plot_fg_and_bg_spectra(fg_ss, bg_ss, index=3000,
                       xscale='log', yscale='log', xlim=(1, None))
