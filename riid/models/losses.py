# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This module contains custom loss functions."""
import numpy as np
from tensorflow.keras import backend as K


def negative_log_f1(y_true: np.ndarray, y_pred: np.ndarray):
    """Implements custom negative log F1 loss score for use in multi-isotope classifiers.

        Args:
            y_true: a list of ground truth.
            y_pred: a list of predictions to compare against the ground truth.

        Returns:
            Returns the custom loss score.

        Raises:
            None
    """
    diff = y_true - y_pred
    negs = K.clip(diff, -1.0, 0.0)
    false_positive = -K.sum(negs, axis=-1)
    true_positive = 1.0 - false_positive
    lower_clip = 1e-20
    true_positive = K.clip(true_positive, lower_clip, 1.0)

    return -K.mean(K.log(true_positive))


def negative_f1(y_true, y_pred) -:
    """Implements custom negative F1 loss score for use in multi-isotope classifiers.

        Args:
            y_true: a list of ground truth.
            y_pred: a list of predictions to compare against the ground truth.

        Returns:
            Returns the custom loss score.

        Raises:
            None
    """
    diff = y_true - y_pred
    negs = K.clip(diff, -1.0, 0.0)
    false_positive = -K.sum(negs, axis=-1)
    true_positive = 1.0 - false_positive
    lower_clip = 1e-20
    true_positive = K.clip(true_positive, lower_clip, 1.0)

    return -K.mean(true_positive)
