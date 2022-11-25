# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to obtain confusion matrices."""
from riid.data.synthetic.static import StaticSynthesizer, get_dummy_sampleset
from riid.models.neural_nets import MLPClassifier
from riid.visualize import confusion_matrix
from riid.data.labeling import BACKGROUND_LABEL

# Generate some dummy seeds
seeds_ss = get_dummy_sampleset(as_seeds=True)
# Separate foreground seeds from background seeds
seeds_labels = seeds_ss.get_labels()
fg_seeds_ss = seeds_ss[seeds_labels != BACKGROUND_LABEL]
bg_seeds_ss = seeds_ss[seeds_labels == BACKGROUND_LABEL]

# Generate some training data
static_syn = StaticSynthesizer(
    snr_function="log10",
    snr_function_args=(.01, 10)
)
train_ss, _, _ = gss.generate(fg_seeds_ss=fg_seeds_ss, bg_seeds_ss=bg_seeds_ss)
train_ss.normalize_sources()
train_ss.normalize()

model = MLPClassifier(
    hidden_layers=(512,),
    dropout=0.25
)
model.fit(train_ss, verbose=0, epochs=200)

# Generate some test data
gss.samples_per_seed = 50
test_ss, _, _ = gss.generate(fg_seeds_ss=fg_seeds_ss, bg_seeds_ss=bg_seeds_ss)
test_ss.normalize_sources()
test_ss.normalize()

# Predict and evaluate
model.predict(test_ss)

fig, ax = confusion_matrix(test_ss, show=True)

# Colormaps can be chosen
fig, ax = confusion_matrix(test_ss, show=True, cmap="Blues")

# additional kwargs for heatmap can be passed as well
fig, ax = confusion_matrix(test_ss, show=True, cmap="RdYlGn", alpha=0.3)


# Values can be displayed as percentage of truth
fig, ax = confusion_matrix(
    test_ss,
    title="Model Performance Accuracy for Each Category",
    show=True,
    cmap="Blues",
    as_percentage=True
)
fig, ax = confusion_matrix(
    test_ss,
    title="Model Performance Accuracy for Each Category",
    show=True,
    cmap="Blues",
    as_percentage=True,
    value_format="0.0%"
)
