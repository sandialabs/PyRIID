# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to use the PyRIID implementations of ARAD.
"""
import numpy as np
import pandas as pd

from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer
from riid.models.neural_nets.arad import ARADv1, ARADv2

# Config
rng = np.random.default_rng(42)
OOD_QUANTILE = 0.99
VERBOSE = False
# Some of the following parameters are set low because this example runs on GitHub Actions and
#   we don't want it taking a bunch of time.
# When running this locally, change the values per their corresponding comment, otherwise
#   the results likely will not be meaningful.
EPOCHS = 5  # Change this to 20+
N_MIXTURES = 50  # Changes this to 1000+
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

# Train the models
results = {}
models = [ARADv1, ARADv2]
for model_class in models:
    arad = model_class()
    model_name = arad.__class__.__name__

    print(f"Training and testing {model_name}...")
    arad.fit(gross_train_ss, epochs=EPOCHS, verbose=VERBOSE)
    arad.predict(gross_train_ss)
    ood_threshold = np.quantile(gross_train_ss.info.recon_error, OOD_QUANTILE)

    reconstructions = arad.predict(test_ss, verbose=VERBOSE)
    ood = test_ss.info.recon_error.values > ood_threshold
    false_positive_rate = ood.mean()
    mean_recon_error = test_ss.info.recon_error.values.mean()

    results[model_name] = {
        "ood_threshold": f"{ood_threshold:.4f}",
        "mean_recon_error": mean_recon_error,
        "false_positive_rate": false_positive_rate,
    }

print(f"Target False Positive Rate: {1-OOD_QUANTILE:.4f}")
print(pd.DataFrame.from_dict(results))
