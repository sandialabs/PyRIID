# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to generate a distance matrix that
   compares every pair of spectra in a SampleSet.
"""
import sys

import matplotlib.pyplot as plt
import seaborn as sns

from riid.data.synthetic import get_dummy_seeds

if len(sys.argv) == 2:
    import matplotlib
    matplotlib.use("Agg")

seeds_ss = get_dummy_seeds(n_channels=16)
distance_df = seeds_ss.get_spectral_distance_matrix()

sns.set(rc={'figure.figsize': (10, 7)})
ax = sns.heatmap(
    distance_df,
    cbar_kws={"label": "Jensen-Shannon Distance"},
    vmin=0,
    vmax=1
)
_ = ax.set_title("Comparing All Seed Pairs Using Jensen-Shannon Distance")
fig = ax.get_figure()
fig.tight_layout()
plt.show()
