# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to train a regressor or classifier branch
from an ARAD latent space.
"""
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error

from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer
from riid.models.neural_nets.arad import ARADv2, ARADLatentPredictor

# Config
rng = np.random.default_rng(42)
VERBOSE = True
# Some of the following parameters are set low because this example runs on GitHub Actions and
#   we don't want it taking a bunch of time.
# When running this locally, change the values per their corresponding comment, otherwise
#   the results likely will not be meaningful.
EPOCHS = 5  # Change this to 20+
N_MIXTURES = 50  # Change this to 1000+
TRAIN_SAMPLES_PER_SEED = 5  # Change this to 20+
TEST_SAMPLES_PER_SEED = 5

# Generate training data
fg_seeds_ss, bg_seeds_ss = get_dummy_seeds(n_channels=128, rng=rng).split_fg_and_bg()
mixed_bg_seed_ss = SeedMixer(bg_seeds_ss, mixture_size=3, rng=rng).generate(N_MIXTURES)
static_synth = StaticSynthesizer(
    samples_per_seed=TRAIN_SAMPLES_PER_SEED,
    snr_function_args=(0, 0),
    return_fg=False,
    return_gross=True,
    rng=rng,
)
_, gross_train_ss = static_synth.generate(fg_seeds_ss[0], mixed_bg_seed_ss)
gross_train_ss.normalize()

# Generate test data
static_synth.samples_per_seed = TEST_SAMPLES_PER_SEED
_, test_ss = static_synth.generate(fg_seeds_ss[0], mixed_bg_seed_ss)
test_ss.normalize()

# Train ARAD model
print("Training ARAD")
arad_v2 = ARADv2()
arad_v2.fit(gross_train_ss, epochs=EPOCHS, verbose=VERBOSE)

# Train regressor to predict SNR
print("Training Regressor")
arad_regressor = ARADLatentPredictor()
_ = arad_regressor.fit(
    arad_v2.model,
    gross_train_ss,
    target_info_columns=["live_time"],
    epochs=10,
    batch_size=5,
    verbose=VERBOSE,
)
regression_predictions = arad_regressor.predict(test_ss)
regression_score = mean_squared_error(gross_train_ss.info.live_time, regression_predictions)
print("Regressor MSE: {:.3f}".format(regression_score))

# Train classifier to predict isotope
print("Training Classifier")
arad_classifier = ARADLatentPredictor(
    loss="categorical_crossentropy",
    metrics=("accuracy", "categorical_crossentropy"),
    final_activation="softmax"
)
arad_classifier.fit(
    arad_v2.model,
    gross_train_ss,
    target_level="Isotope",
    epochs=10,
    batch_size=5,
    verbose=VERBOSE,
)
arad_classifier.predict(test_ss)
classification_score = f1_score(test_ss.get_labels(), test_ss.get_predictions(), average="micro")
print("Classification F1 Score: {:.3f}".format(classification_score))
