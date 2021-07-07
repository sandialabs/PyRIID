# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This example demonstrates a comparison between Poisson Bayes and MLP classifiers."""
from time import time

from riid.data import load_seeds
from riid.models.bayes import PoissonBayes
from riid.synthetic import GammaSpectraSynthesizer
from riid.visualize import plot_live_time_vs_snr, plot_strength_vs_score
from sklearn.metrics import f1_score

seeds = load_seeds("gm_561", "measured", "seven_isotopes.smpl")
# Separate foreground seeds from background seeds
seeds.clip_negatives()
fg_seeds_ss = seeds.get_indices(seeds.labels != "background")
bg_seeds_ss = seeds.get_indices(seeds.labels == "background")
# Create a data synthesizer and generate some test data
gss = GammaSpectraSynthesizer(
    seeds,
    subtract_background=False,
    # log10 sampling samples lower SNR values more frequently.
    # This makes the SampleSet overall "harder" to classify.
    snr_function="log10",
    samples_per_seed=100
)
test_ss = gss.generate()

# Create model
model = PoissonBayes(fg_seeds_ss)

# Predict
tstart = time()
model.predict(test_ss, bg_seeds_ss, normalize_scores=True)
delay = time() - tstart

score = f1_score(test_ss.labels, test_ss.predictions, average="micro")
print("F1 Score: {:.3f}".format(score))
print("Delay:    {:.2f}s".format(delay))

plot_live_time_vs_snr(test_ss)
plot_strength_vs_score(test_ss, ylim=(None, None))
