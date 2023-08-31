# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates a comparison between Poisson Bayes and MLP classifiers."""
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer
from riid.models.bayes import PoissonBayesClassifier
from riid.metrics import precision_recall_curve
from riid.models.neural_nets import MLPClassifier
from riid.visualize import plot_precision_recall

if len(sys.argv) == 2:
    import matplotlib
    matplotlib.use("Agg")

# Generate some training data
fg_seeds_ss, bg_seeds_ss = get_dummy_seeds(n_channels=64).split_fg_and_bg()


mixed_bg_seed_ss = SeedMixer(bg_seeds_ss, mixture_size=3).generate(10)

static_synth = StaticSynthesizer(
    samples_per_seed=100,
    live_time_function_args=(1, 10),
    snr_function="log10",
    snr_function_args=(1, 20),
    return_fg=True,
    return_gross=True,
)
train_fg_ss, _ = static_synth.generate(fg_seeds_ss, mixed_bg_seed_ss, verbose=False)
train_fg_ss.normalize()

model_nn = MLPClassifier(hidden_layers=(5,))
model_nn.fit(train_fg_ss, epochs=10, patience=5, verbose=1)

# Create PB model
model_pb = PoissonBayesClassifier()
model_pb.fit(fg_seeds_ss)

# Generate some test data
static_synth.samples_per_seed = 50
test_fg_ss, test_gross_ss = static_synth.generate(fg_seeds_ss, mixed_bg_seed_ss,
                                                  verbose=False)
test_bg_ss = test_gross_ss - test_fg_ss
test_fg_ss.normalize()
test_gross_ss.sources.drop(bg_seeds_ss.sources.columns, axis=1, inplace=True)
test_gross_ss.normalize_sources()

# Plot
fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
results = {}
for model, tag, ax in zip([model_nn, model_pb], ["NN", "PB"], axs):
    if tag == "NN":
        labels = test_fg_ss.get_labels()
        model.predict(test_fg_ss)
        predictions = test_fg_ss.get_predictions()
        precision, recall, thresholds = precision_recall_curve(test_fg_ss)
    elif tag == "PB":
        labels = test_gross_ss.get_labels()
        model.predict(test_gross_ss, test_bg_ss)
        predictions = test_gross_ss.get_predictions()
        precision, recall, thresholds = precision_recall_curve(test_gross_ss)
    else:
        raise ValueError()

    score = f1_score(labels, predictions, average="weighted")
    print(f"{tag} F1-score: {score:.3f}")

    plot_precision_recall(
        precision=precision,
        recall=recall,
        title=f"{tag}\nPrecision VS Recall",
        fig_ax=(fig, ax),
        show=False,
    )

plt.show()
