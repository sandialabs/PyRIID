# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to use the MLP classifier."""
import numpy as np
from sklearn.metrics import f1_score

from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer
from riid.models.neural_nets import MLPClassifier

# Generate some training data
fg_seeds_ss, bg_seeds_ss = get_dummy_seeds().split_fg_and_bg()
mixed_bg_seed_ss = SeedMixer(bg_seeds_ss, mixture_size=3).generate(1)

static_synth = StaticSynthesizer(
    samples_per_seed=100,
    snr_function="log10",
    return_fg=False,
    return_gross=True,
)
_, train_ss = static_synth.generate(fg_seeds_ss, mixed_bg_seed_ss)
train_ss.normalize()

model = MLPClassifier()
model.fit(train_ss, epochs=10, patience=5)

# Generate some test data
static_synth.samples_per_seed = 50
_, test_ss = static_synth.generate(fg_seeds_ss, mixed_bg_seed_ss)
test_ss.normalize()

# Predict
model.predict(test_ss)

score = f1_score(test_ss.get_labels(), test_ss.get_predictions(), average="micro")
print("F1 Score: {:.3f}".format(score))

# Get confidences
confidences = test_ss.get_confidences(
    fg_seeds_ss,
    bg_seed_ss=mixed_bg_seed_ss,
    bg_cps=300
)
print(f"Avg Confidence: {np.mean(confidences):.3f}")
