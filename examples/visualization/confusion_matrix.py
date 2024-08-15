# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to obtain confusion matrices."""
import sys

from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer
from riid.models.neural_nets import MLPClassifier
from riid.visualize import confusion_matrix

if len(sys.argv) == 2:
    import matplotlib
    matplotlib.use("Agg")

SYNTHETIC_DATA_CONFIG = {
    "snr_function": "log10",
    "snr_function_args": (.01, 10),
}
fg_seeds_ss, bg_seeds_ss = get_dummy_seeds().split_fg_and_bg()
mixed_bg_seed_ss = SeedMixer(bg_seeds_ss, mixture_size=3)\
    .generate(1)

train_ss, _ = StaticSynthesizer(**SYNTHETIC_DATA_CONFIG)\
    .generate(fg_seeds_ss, mixed_bg_seed_ss)
train_ss.normalize()

model = MLPClassifier()
model.fit(train_ss, verbose=0, epochs=50)

# Generate some test data
SYNTHETIC_DATA_CONFIG = {
    "snr_function": "log10",
    "snr_function_args": (.01, 10),
    "samples_per_seed": 50,
}

test_ss, _ = StaticSynthesizer(**SYNTHETIC_DATA_CONFIG)\
    .generate(fg_seeds_ss, mixed_bg_seed_ss)
test_ss.normalize()

# Predict and evaluate
model.predict(test_ss)

fig, ax = confusion_matrix(test_ss, show=True)
