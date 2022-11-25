# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to use the MLP classifier."""
from time import time

from riid.models.neural_nets import MLPClassifier
from riid.data.labeling import BACKGROUND_LABEL
from riid.data.synthetic.static import StaticSynthesizer, get_dummy_sampleset
from sklearn.metrics import f1_score

# Generate some training data
seeds_ss = get_dummy_sampleset(as_seeds=True)
seeds_labels = seeds_ss.get_labels()

fg_seeds_ss = seeds_ss[seeds_labels != BACKGROUND_LABEL]
fg_seeds_ss.sources.drop(BACKGROUND_LABEL, axis=1, level="Category", inplace=True)
bg_seeds_ss = seeds_ss[seeds_labels == BACKGROUND_LABEL]

static_syn = StaticSynthesizer(
    samples_per_seed=500,
    # log10 sampling samples lower SNR values more frequently.
    # This makes the SampleSet overall "harder" to classify.
    snr_function="log10"
)
_, _, train_ss = gss.generate(fg_seeds_ss=fg_seeds_ss, bg_seeds_ss=bg_seeds_ss)
train_ss.normalize_sources()
train_ss.normalize()
model = MLPClassifier()
model.fit(train_ss, verbose=1, epochs=200, patience=20)

# Generate some test data
gss.samples_per_seed = 100
_, _, test_ss = gss.generate(fg_seeds_ss=fg_seeds_ss, bg_seeds_ss=bg_seeds_ss)
test_ss.normalize_sources()
test_ss.normalize()
# Predict
tstart = time()
model.predict(test_ss)
delay = time() - tstart

score = f1_score(test_ss.get_labels(), test_ss.get_predictions(), average="micro")
print("F1 Score: {:.3f}".format(score))
print("Delay:    {:.2f}s".format(delay))
