# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This module provides custom metrics."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def multi_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tf.Tensor:
    """This metric provides a measure of the F1 score of two tensors.

        y_true and y_pred are assumed to sum to 1.

        Args:
            y_true: a list of ground truth.
            y_pred: a list of predictions to compare against the ground truth.

        Returns:
            Returns the custom loss score.

        Raises:
            None.
    """
    diff = y_true - y_pred
    negs = K.clip(diff, -1.0, 0.0)
    false_positive = -K.sum(negs, axis=-1)
    true_positive = 1.0 - false_positive

    return K.mean(true_positive)


def single_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tf.Tensor:
    """ Computes the weighted F1 score for the maximum prediction and maximum
    ground truth.

        y_true and y_pred are assumed to sum to 1.

        Args:
            y_true: a list of ground truth.
            y_pred: a list of predictions to compare against the ground truth.

        Returns:
            Returns the custom loss score.

        Raises:
            None
    """
    a = tf.dtypes.cast(y_true == K.max(y_true, axis=1)[:, None], tf.float32)
    b = tf.dtypes.cast(y_pred == K.max(y_pred, axis=1)[:, None], tf.float32)

    TP_mat = tf.dtypes.cast(K.all(tf.stack([a, b]), axis=0), tf.float32)
    FP_mat = tf.dtypes.cast(K.all(tf.stack([a != b, b == 1]), axis=0), tf.float32)
    FN_mat = tf.dtypes.cast(K.all(tf.stack([a != b, a == 1]), axis=0), tf.float32)

    TPs = K.sum(TP_mat, axis=0)
    FPs = K.sum(FP_mat, axis=0)
    FNs = K.sum(FN_mat, axis=0)

    F1s = 2 * TPs / (2*TPs + FNs + FPs + tf.fill(tf.shape(TPs), K.epsilon()))

    support = K.sum(a, axis=0)
    f1 = K.sum(F1s * support) / K.sum(support)
    return f1
