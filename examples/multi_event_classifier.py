# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This example demonstrates how to use the MLP classifier."""
from copy import deepcopy as copy

from riid.data.labeling import BACKGROUND_LABEL
from riid.data.synthetic.static import StaticSynthesizer, get_dummy_sampleset
from riid.models.neural_nets import MLPClassifier, MultiEventClassifier
from sklearn.metrics import f1_score

seeds_ss = get_dummy_sampleset(as_seeds=True)
gss = StaticSynthesizer(
    # log10 sampling samples lower SNR values more frequently.
    # This makes the SampleSet overall "harder" to classify.
    snr_function="log10",
    samples_per_seed=10
)

ss_labels = seeds_ss.get_labels()
fg_seeds_ss = seeds_ss[ss_labels != BACKGROUND_LABEL]
bg_seeds_ss = seeds_ss[ss_labels == BACKGROUND_LABEL]

# Generate some training data
train_ss, train_bg, _ = gss.generate(fg_seeds_ss, bg_seeds_ss)

# Train two single event classifiers
model1 = MLPClassifier()
model1.fit(train_ss, bg_ss=train_bg, verbose=1, epochs=200, patience=20)

model2 = MLPClassifier()
model2.fit(train_ss, bg_ss=train_bg, verbose=1, epochs=200, patience=20)

# Generate two sample sets (with same sources but predictions from different models)
train2a_ss, train2a_bg_ss, _ = gss.generate(fg_seeds_ss, bg_seeds_ss)
train2b_ss = copy(train2a_ss)
train2b_bg_ss = copy(train2a_bg_ss)

model1.predict(train2a_ss, train2a_bg_ss)
model2.predict(train2b_ss, train2b_bg_ss)

# Train MultiEvent model
mec = MultiEventClassifier()
mec.fit([train2a_ss, train2b_ss], train2a_ss.get_source_contributions())

# Make predictions on multi model

multi_preds = mec.predict([train2a_ss, train2b_ss])

# Compare performance for single event models and multi-event model
m1_f1_score = f1_score(train2a_ss.get_predictions(),
                       train2a_ss.get_labels(),
                       average="weighted")
m2_f1_score = f1_score(train2b_ss.get_predictions(),
                       train2b_ss.get_labels(),
                       average="weighted")

multi_f1_score = f1_score(multi_preds.values.argmax(axis=1),
                          train2a_ss.get_source_contributions().values.argmax(axis=1),
                          average="weighted")

print_str = f"M1 F1 Score: {m1_f1_score:.2f}"
print_str += f"\nM2 F1 Score: {m2_f1_score:.2f}"
print_str += f"\nM12 F1 Score: {multi_f1_score:.2f}\n"
print(print_str)
