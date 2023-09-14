"""This example demonstrates how to train the Label Proportion
Estimator with a semi-supervised loss function."""
import os

from sklearn.metrics import mean_absolute_error

from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer
from riid.models.neural_nets import LabelProportionEstimator

# Generate some mixture training data.
fg_seeds_ss, bg_seeds_ss = get_dummy_seeds().split_fg_and_bg()
mixed_bg_seeds_ss = SeedMixer(
    bg_seeds_ss,
    mixture_size=3,
    dirichlet_alpha=2
).generate(100)

static_syn = StaticSynthesizer(
    samples_per_seed=100,
    bg_cps=300.0,
    live_time_function_args=(60, 600),
    snr_function_args=(0, 0),
    return_fg=False,
    return_gross=True,
)

_, bg_ss = static_syn.generate(fg_seeds_ss[0], mixed_bg_seeds_ss)
bg_ss.drop_sources_columns_with_all_zeros()
bg_ss.normalize()

# Create the model
model = LabelProportionEstimator(
    hidden_layers=(64,),
    # The supervised loss can either be "sparsemax"
    # or "categorical_crossentropy".
    sup_loss="categorical_crossentropy",
    # The unsupervised loss be "poisson_nll", "normal_nll",
    # "sse", or "weighted_sse".
    unsup_loss="poisson_nll",
    # This controls the tradeoff between the sup
    # and unsup losses.,
    beta=1e-4,
    optimizer="RMSprop",
    learning_rate=1e-2,
    hidden_layer_activation="relu",
    l2_alpha=1e-4,
    dropout=0.05,
)

# Train the model.
model.fit(
    bg_seeds_ss,
    bg_ss,
    batch_size=10,
    epochs=10,
    validation_split=0.2,
    verbose=True,
    bg_cps=300
)

# Generate some test data.
static_syn.samples_per_seed = 50
_, test_bg_ss = static_syn.generate(fg_seeds_ss[0], mixed_bg_seeds_ss)
test_bg_ss.normalize(p=1)
test_bg_ss.drop_sources_columns_with_all_zeros()

model.predict(test_bg_ss)

test_meas = mean_absolute_error(
    test_bg_ss.sources.values,
    test_bg_ss.prediction_probas.values
)
print(f"Mean Test MAE: {test_meas.mean():.3f}")

# Save model in ONNX format
model_info_path, model_path = model.save("./model.onnx")

loaded_model = LabelProportionEstimator()
loaded_model.load(model_path)

loaded_model.predict(test_bg_ss)
test_maes = mean_absolute_error(
    test_bg_ss.sources.values,
    test_bg_ss.prediction_probas.values
)

print(f"Mean Test MAE: {test_maes.mean():.3f}")

# Clean up model file - remove this if you want to keep the model
os.remove(model_info_path)
os.remove(model_path)
