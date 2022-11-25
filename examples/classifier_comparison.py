# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates a comparison between Poisson Bayes and MLP classifiers."""
from time import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

from riid.data.labeling import BACKGROUND_LABEL
from riid.data.synthetic.static import StaticSynthesizer, get_dummy_sampleset
from riid.models.bayes import PoissonBayes
from riid.models.metrics import precision_recall_curve
from riid.models.neural_nets import MLPClassifier
from riid.visualize import plot_precision_recall

seeds_ss = get_dummy_sampleset(as_seeds=True)
# Separate foreground seeds from background seeds
seeds_ss.clip_negatives()
seeds_ss.normalize()
seeds_labels = seeds_ss.get_labels()
fg_seeds_ss = seeds_ss[seeds_labels != BACKGROUND_LABEL]
fg_seeds_ss.sources.drop(BACKGROUND_LABEL, axis=1, level="Category", inplace=True)
bg_seeds_ss = seeds_ss[seeds_labels == BACKGROUND_LABEL]
# Create a data synthesizer and generate some test data
gss = StaticSynthesizer(
    samples_per_seed=500,
    # log10 sampling samples lower SNR values more frequently.
    # This makes the SampleSet overall "harder" to classify.
    snr_function="log10",
)
_, train_bg_ss, train_ss = gss.generate(fg_seeds_ss=fg_seeds_ss, bg_seeds_ss=bg_seeds_ss)

gss.samples_per_seed = 100
_, test_bg_ss, test_ss = gss.generate(fg_seeds_ss=fg_seeds_ss, bg_seeds_ss=bg_seeds_ss)

# Create NN Model
model_nn = MLPClassifier()
# Currently, call to fit() breaks on the use of x_train.shape[1] and y_train.shape[1] as
# size is flattened to (512, ) for both.
# Fixing the use of shape to access shape[0] resolves this, but ultimately the call to
# fit() still fails because the flattened data arrays do not work as inputs for model.fit,
# and the following error results:
#  ValueError: Data cardinality is ambiguous:
#    x sizes: 409, 409
#    y sizes: 8
model_nn.fit(ss=train_ss, bg_ss=train_bg_ss)

# Create PB model
model_pb = PoissonBayes(fg_seeds_ss)
# Make the bg_seed into the bg expected

fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

results = {}
for model, tag, ax in zip([model_nn, model_pb], ["Neural Network", "PoissonBayes"], axs):
    tstart = time()
    if tag == "PoissonBayes":
        model.predict(test_ss)
    elif tag == "Neural Network":
        model.predict(test_ss, test_bg_ss)
    delay = time() - tstart
    labels = test_ss.get_labels()
    predictions = test_ss.get_predictions()
    score = f1_score(labels, predictions, average="micro")

    results[tag] = {
        "F1": score,
        "Accuracy": (labels == predictions).mean(),
        "Execution time (sec)": delay
    }

    # precision recall curve
    precision, recall, thresholds = precision_recall_curve(test_ss)
    plot_precision_recall(
        precision=precision,
        recall=recall,
        title=f"{tag}\nPrecision VS Recall",
        fig_ax=(fig, ax),
        show=False,
    )

print("\nRESULTS")
print("-" * 50)
print(pd.DataFrame(results))

plt.show()
