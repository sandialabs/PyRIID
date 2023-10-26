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
from riid.models.neural_nets.arad import ARAD, ARADv1TF, ARADv2TF

# Config
rng = np.random.default_rng(42)
OOD_QUANTILE = 0.99
VERBOSE = True
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

# Train the models
print("Training ARADv1...")
arad_v1 = ARAD(model=ARADv1TF())
arad_v1.fit(gross_train_ss, epochs=EPOCHS, verbose=VERBOSE)
arad_v1.predict(gross_train_ss)
v1_ood_threshold = np.quantile(gross_train_ss.info.recon_error, OOD_QUANTILE)

print("Training ARADv2...")
arad_v2 = ARAD(model=ARADv2TF())
arad_v2.fit(gross_train_ss, epochs=EPOCHS, verbose=VERBOSE)
arad_v2.predict(gross_train_ss)
v2_ood_threshold = np.quantile(gross_train_ss.info.recon_error, OOD_QUANTILE)

# Generate test data
static_synth.samples_per_seed = TEST_SAMPLES_PER_SEED
_, test_ss = static_synth.generate(fg_seeds_ss[0], mixed_bg_seed_ss)
test_ss.normalize()

# Predict

arad_v1_reconstructions = arad_v1.predict(test_ss, verbose=True)
arad_v1_ood = test_ss.info.recon_error.values > v1_ood_threshold
arad_v1_false_positive_rate = arad_v1_ood.mean()
arad_v1_mean_recon_error = test_ss.info.recon_error.values.mean()

arad_v2_reconstructions = arad_v2.predict(test_ss, verbose=True)
arad_v2_ood = test_ss.info.recon_error.values > v2_ood_threshold
arad_v2_false_positive_rate = arad_v2_ood.mean()
arad_v2_mean_recon_error = test_ss.info.recon_error.values.mean()

results = {
    "ARADv1": {
        "ood_threshold": f"KLD={v1_ood_threshold:.4f}",
        "mean_recon_error": arad_v1_mean_recon_error,
        "false_positive_rate": arad_v1_false_positive_rate,
    },
    "ARADv2": {
        "ood_threshold": f"JSD={v2_ood_threshold:.4f}",
        "mean_recon_error": arad_v2_mean_recon_error,
        "false_positive_rate": arad_v2_false_positive_rate,
    }
}
print(f"Target False Positive Rate: {1-OOD_QUANTILE:.4f}")
print(pd.DataFrame.from_dict(results))
