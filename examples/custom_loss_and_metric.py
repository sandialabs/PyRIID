# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This example demonstrates custom loss functions and metrics."""
import tensorflow as tf
from riid.models.losses import negative_log_f1
from riid.models.metrics import multi_f1

y_true = tf.constant([.524, .175, .1, .1, 0, .1])
y_pred = tf.constant([.2, .2, .2, .1, .2, .1])
f1 = multi_f1(y_true, y_pred)
loss = negative_log_f1(y_true, y_pred)
print(f"F1 Score: {f1:.3f}")
print(f"Loss:     {loss:.3f}")
