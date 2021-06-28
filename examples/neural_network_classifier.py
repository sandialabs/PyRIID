# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This example demonstrates how to use the MLP classifier."""
from time import time

from riid.data import load_seeds
from riid.models.neural_nets import MLPClassifier
from riid.synthetic import GammaSpectraSynthesizer
from riid.visualize import plot_live_time_vs_snr, plot_strength_vs_score
from sklearn.metrics import f1_score

# Generate some training data
seeds = load_seeds("gm_561", "measured", "seven_isotopes.smpl")
gss = GammaSpectraSynthesizer(
    seeds,
    subtract_background=False,
    # log10 sampling samples lower SNR values more frequently.
    # This makes the SampleSet overall "harder" to classify.
    snr_function="log10",
    samples_per_seed=500
)
ss_train = gss.generate()

model = MLPClassifier(
    ss_train.n_channels,
    ss_train.label_matrix.shape[1]
)
model.fit(ss_train, verbose=0, epochs=200, patience=20)

# Generate some test data
gss.samples_per_seed = 100
ss_test = gss.generate()

# Predict
tstart = time()
model.predict(ss_test)
delay = time() - tstart

score = f1_score(ss_test.labels, ss_test.predictions, average="micro")
print("F1 Score: {:.3f}".format(score))
print("Delay:    {:.2f}s".format(delay))

plot_live_time_vs_snr(ss_test)
plot_strength_vs_score(ss_test)
