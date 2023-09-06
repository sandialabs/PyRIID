# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module provides custom model metrics."""
import numpy as np
import sklearn

from riid.data.sampleset import SampleSet


def multi_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate a measure of the F1 score of two tensors.

    Values for `y_true` and `y_pred` are assumed to sum to 1.

    Args:
        y_true: list of ground truth
        y_pred: list of predictions to compare against the ground truth

    Returns:
        Multi F1-score value(s)
    """
    from keras import backend as K

    diff = y_true - y_pred
    negs = K.clip(diff, -1.0, 0.0)
    false_positive = -K.sum(negs, axis=-1)
    true_positive = 1.0 - false_positive

    return K.mean(true_positive)


def single_f1(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute the weighted F1 score for the maximum prediction and maximum ground truth.

    Values for `y_true` and `y_pred` are assumed to sum to 1.

    Args:
        y_true: list of ground truth
        y_pred: list of predictions to compare against the ground truth

    Returns:
        F1-score value(s)
    """
    import tensorflow as tf
    from keras import backend as K

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


def harmonic_mean(x, y):
    """Compute the harmonic mean of two same-dimensional arrays.

    Used to compute F1 score:

    ```
    f1_score = harmonic_mean(precision, recall)
    ```

    Args:
        x (array-like): numeric or array_like of numerics
        y (array-like): numeric or array_like of numerics matching the shape/type of `x`

    Returns:
        Array-like harmonic mean of `x` and `y`
    """
    return 2 * x * y / (x + y)


def precision_recall_curve(ss: SampleSet, smooth: bool = True, multiclass: bool = None,
                           include_micro: bool = True, target_level: str = "Isotope",
                           minimum_contribution: float = 0.01):
    """Similar to `sklearn.metrics.precision_recall_curve`, however, this function
    computes the precision and recall for each class, and supports both multi-class
    and multi-label problems.

    The reason this is necessary is that in multi-class problems, for a single sample,
    all predictions are discarded except for the argmax.

    Args:
        ss: `SampleSet` that predictions were generated on
        smooth: if True, precision is smoothing is applied to make a monotonically
            decreasing precision function
        multiclass: set to True if this is a multi-class (i.e. y_true is one-hot) as
            opposed to multi-label (i.e. labels are not mutually exclusive). Ff True,
            predictions will be masked such that non-argmax predictions are set to zero
            (this prevents inflating the precision by continuing past a point that could
            be pragmatically useful). Furthermore, in the multiclass case the recall is
            not guaranteed to reach 1.0.
        include_micro: if True, compute an additional precision and recall for the
            micro-average across all labels and put it under entry `"micro"`
        target_level: `SampleSet.sources` and `SampleSet.prediction_probas` column level to use
        minimum_contribution: threshold for a source to be considered a ground truth positive
            label. if this is set to `None` the raw mixture ratios will be used as y_true.

    Returns:
        precision (dict): dict with keys for each label and values that are the
            monotonically increasing precision values at each threshold
        recall (dict): dict with keys for each label and values that are the
            monotonically decreasing recall values at each threshold
        thresholds (dict): dict with keys for each label and values that are the
            monotonically increasing thresholds on the decision function used to compute
            precision and recall

    References:
        - [Precision smoothing](
           https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

    """
    y_true = ss.sources.groupby(axis=1, level=target_level).sum()
    if minimum_contribution is not None:
        y_true = (y_true > minimum_contribution).astype(int)
    y_pred = ss.prediction_probas.groupby(axis=1, level=target_level).sum()

    # switch from pandas to numpy
    labels = y_true.columns
    n_classes = len(labels)
    y_true = y_true.values.copy()
    y_pred = y_pred.values.copy()

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch between truth and predictions, "
            f"{y_true.shape} != {y_pred.shape}. "
            f"It is possible that the `target_level` is incorrect."
        )

    if multiclass is None:
        # infer whether multi-class or multi-label
        multiclass = not np.any(y_true.sum(axis=1) != 1)

    # drop nans
    notnan = ~np.isnan(y_pred).any(axis=1)
    y_true = y_true[notnan, :]
    y_pred = y_pred[notnan, :]

    y_pred_min = None
    if multiclass:
        # shift predictions to force positive
        if np.any(y_pred < 0):
            y_pred_min = y_pred.min()
            y_pred -= y_pred_min

        # mask the predictions by argmax
        pred_mask = np.eye(n_classes)[np.argmax(y_pred, axis=1)]
        # mask the predictions
        y_pred *= pred_mask

    precision = dict()
    recall = dict()
    thresholds = dict()
    for i, label in enumerate(labels):
        precision[label], recall[label], thresholds[label] = _pr_curve(
            y_true[:, i], y_pred[:, i], multiclass=multiclass, smooth=smooth
        )

    if include_micro:
        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], thresholds["micro"] = _pr_curve(
            y_true.ravel(), y_pred.ravel(), multiclass=multiclass, smooth=smooth
        )

    # un-shift thresholds if predictions were shifted
    if y_pred_min is not None:
        thresholds = {k: v + y_pred_min for k, v in thresholds.items()}

    return precision, recall, thresholds


def average_precision_score(precision, recall):
    """Compute the average precision (area under the curve) for each precision/recall
    pair.

    Args:
        precision (dict): return value of `ctutil.evaluation.precision_recall_curve()`
        recall (dict): return value of `ctutil.evaluation.precision_recall_curve()`

    Returns:
        (dict): average precision values (float) for each label in precision/recall
    """
    return {label: _integrate(recall[label], precision[label]) for label in recall}


def _step(x):
    """Compute the right going maximum of `x` and all previous values of `x`.

    Args:
        x (array-like): 1D array to process

    Returns:
        (array-like): right-going maximum of `x`

    """
    y = np.array(x)
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1])
    return y


def _integrate(x, y, y_left=True):
    """Integrate an (x, y) function pair.

    Args:
        x (array-like): 1D array of x values
        y (array-like): 1D array of y values
        y_left: if true, omit the last value of y, else, omit the first value

    Returns:
        (float): integrated "area under the curve"

    """
    delta_x = x[1:] - x[:-1]
    y_trimmed = y[:-1] if y_left else y[1:]
    return np.abs(np.sum(delta_x * y_trimmed))


def _pr_curve(y_true, y_pred, multiclass, smooth):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
        y_true, y_pred
    )

    if smooth:
        precision = _step(precision)

    if multiclass:
        # remove the point where threshold=0 and recall=1
        precision = precision[1:]
        recall = recall[1:]
        thresholds = thresholds[1:]

    return precision, recall, thresholds


def build_keras_semisupervised_metric_func(keras_metric_func, activation_func,
                                           n_labels):
    def metric_func(y_true, y_pred):
        return keras_metric_func(y_true[:, :n_labels], activation_func(y_pred))
    metric_func.__name__ = keras_metric_func.__name__

    return metric_func
