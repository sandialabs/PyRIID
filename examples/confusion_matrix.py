# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This example demonstrates how to obtain confusion matrices."""
from riid.data import load_seeds
from riid.models.neural_nets import MLPClassifier
from riid.synthetic import GammaSpectraSynthesizer
from riid.visualize import confusion_matrix

# Generate some training data
seeds = load_seeds("gm_561", "measured", "seven_isotopes.smpl")
gss = GammaSpectraSynthesizer(seeds, snr_function="log10", snr_function_args=(.01,10))
train_ss = gss.generate()

model = MLPClassifier(
    train_ss.n_channels,
    train_ss.label_matrix.shape[1],
    hidden_layers=(512,),
    dropout=0.25
)
model.fit(train_ss, verbose=0, epochs=200)

# Generate some test data
gss.samples_per_seed = 50
test_ss = gss.generate()

# Predict and evaluate
model.predict(test_ss)

fig, ax = confusion_matrix(test_ss, show=True)

# Colormaps can be chosen
fig, ax = confusion_matrix(test_ss, show=True, cmap="Blues")

# additional kwargs for heatmap can be passed as well
fig, ax = confusion_matrix(test_ss, show=True, cmap="RdYlGn", fill_opacity=0.3)


# Values can be displayed as percentage of truth
fig, ax = confusion_matrix(test_ss, title="Model Performance Accuracy for Each Category", show=True, cmap="Blues", as_percentage=True)
fig, ax = confusion_matrix(test_ss, title="Model Performance Accuracy for Each Category", show=True, cmap="Blues", as_percentage=True, value_format="0.0%")

