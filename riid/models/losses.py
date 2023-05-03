# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains custom loss functions."""
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from math import pi


def negative_log_f1(y_true: np.ndarray, y_pred: np.ndarray):
    """Implements custom negative log F1 loss score for use in multi-isotope classifiers.

    Args:
        y_true: Defines a list of ground truth.
        y_pred: Defines a list of predictions to compare against the ground truth.

    Returns:
        The custom loss score on a log scale.

    """
    diff = y_true - y_pred
    negs = K.clip(diff, -1.0, 0.0)
    false_positive = -K.sum(negs, axis=-1)
    true_positive = 1.0 - false_positive
    lower_clip = 1e-20
    true_positive = K.clip(true_positive, lower_clip, 1.0)

    return -K.mean(K.log(true_positive))


def negative_f1(y_true, y_pred):
    """Implements custom negative F1 loss score for use in multi-isotope classifiers.

    Args:
        y_true: Defines a list of ground truth.
        y_pred: Defines a list of predictions to compare against the ground truth.

    Returns:
        The custom loss score.

    """
    diff = y_true - y_pred
    negs = K.clip(diff, -1.0, 0.0)
    false_positive = -K.sum(negs, axis=-1)
    true_positive = 1.0 - false_positive
    lower_clip = 1e-20
    true_positive = K.clip(true_positive, lower_clip, 1.0)

    return -K.mean(true_positive)


def build_semisupervised_loss_func(supervised_loss_func, unsupervised_loss_func,
                                   dictionary, beta):
    def _semisupervised_loss_func(spectra, y_true, y_logits, y_lpes):
        sup_losses = supervised_loss_func(y_true, y_logits)
        unsup_losses = reconstruction_error(spectra, y_lpes, dictionary,
                                            unsupervised_loss_func)
        semisup_losses = sup_losses + beta * unsup_losses
        return sup_losses, unsup_losses, semisup_losses

    return _semisupervised_loss_func


def sse_diff(spectra, reconstructed_spectra):
    """Computes the sum of squares error.

    TODO: refactor to assume spectral inputs are in the same form

    Args:
        spectra: the spectral samples, assumed to be in counts
        reconstructed_spectra: the reconstructed spectra created using a
            dictionary with label proportion estimates
    """
    total_counts = tf.reduce_sum(spectra, axis=-1)
    normalized_spectra = tf.divide(
        spectra,
        tf.reshape(total_counts, (-1, 1))
    )
    diff = normalized_spectra - reconstructed_spectra
    norm_diff = tf.norm(diff, axis=-1)
    squared_norm_diff = tf.square(norm_diff)
    return squared_norm_diff


def poisson_nll_diff(spectra, reconstructed_spectra, eps=1e-8):
    """Computes the Poisson Negative Log-Likelihood.

    TODO: refactor to assume spectral inputs are in the same form

    Args:
        spectra: the spectral samples, assumed to be in counts
        reconstructed_spectra: the reconstructed spectra created using a
            dictionary with label proportion estimates
    """
    total_counts = tf.reduce_sum(spectra, axis=-1)
    scaled_reconstructed_spectra = tf.multiply(
        reconstructed_spectra,
        tf.reshape(total_counts, (-1, 1))
    )
    log_reconstructed_spectra = tf.math.log(scaled_reconstructed_spectra + eps)
    diff = tf.nn.log_poisson_loss(
        spectra,
        log_reconstructed_spectra,
        compute_full_loss=True
    )
    diff = tf.reduce_sum(diff, axis=-1)

    return diff


def normal_nll_diff(spectra, reconstructed_spectra, eps=1e-8):
    """Computes the Normal Negative Log-Likelihood.

    TODO: refactor to assume spectral inputs are in the same form

    Args:
        spectra: the spectral samples, assumed to be in counts
        reconstructed_spectra: the reconstructed spectra created using a
            dictionary with label proportion estimates
    """
    total_counts = tf.reduce_sum(spectra, axis=-1)
    scaled_reconstructed_spectra = tf.multiply(
        reconstructed_spectra,
        tf.reshape(total_counts, (-1, 1))
    )

    var = tf.clip_by_value(spectra, clip_value_min=1, clip_value_max=np.inf)

    sigma_term = tf.math.log(2 * pi * var)
    mu_term = tf.math.divide(tf.math.square(scaled_reconstructed_spectra - spectra), var)
    diff = sigma_term + mu_term
    diff = 0.5 * tf.reduce_sum(diff, axis=-1)

    return diff


def weighted_sse_diff(spectra, reconstructed_spectra):
    """ Computes the Normal Negative Log-Likelihood under constant variance
    (this reduces to the SSE, just on a different scale).

    Args:
        spectra: the spectral samples, assumed to be in counts
        reconstructed_spectra: the reconstructed spectra created using a
            dictionary with label proportion estimates
    """
    total_counts = tf.reduce_sum(spectra, axis=1)
    scaled_reconstructed_spectra = tf.multiply(
        reconstructed_spectra,
        tf.reshape(total_counts, (-1, 1))
    )

    sample_variance = tf.sqrt(tf.math.reduce_variance(spectra, axis=1))

    sigma_term = tf.math.log(2 * pi * sample_variance)

    mu_term = tf.math.divide(
        tf.math.square(scaled_reconstructed_spectra - spectra),
        tf.reshape(sample_variance, (-1, 1))
    )
    diff = 0.5 * (sigma_term + tf.reduce_sum(mu_term, axis=-1))

    return diff


def reconstruction_error(spectra, lpes, dictionary, diff_func):
    reconstructed_spectra = tf.matmul(lpes, dictionary)
    reconstruction_errors = diff_func(spectra, reconstructed_spectra)
    return reconstruction_errors


# based off code from Tensorflow-Addons (https://www.tensorflow.org/addons)
def sparsemax(logits, axis: int = -1) -> tf.Tensor:
    """Sparsemax activation function.

    Args:
        logits: tensor of logits (should not be activated)
        axis: axis along which activation is applied
    """

    logits = tf.convert_to_tensor(logits, name="logits")

    shape = logits.get_shape()
    rank = shape.rank
    is_last_axis = (axis == -1) or (axis == rank - 1)

    if not is_last_axis:
        raise ValueError("Currently only last axis is supported.")

    output = _compute_2d_sparsemax(logits)
    output.set_shape(shape)
    return output


# based off code from Tensorflow-Addons (https://www.tensorflow.org/addons)
@tf.function
def sparsemax_loss_from_logits(y_true, logits_pred) -> tf.Tensor:
    logits = tf.convert_to_tensor(logits_pred, name="logits")
    sparsemax_values = tf.convert_to_tensor(sparsemax(logits_pred), name="sparsemax")
    labels = tf.convert_to_tensor(y_true, name="labels")

    z = logits
    sum_s = tf.where(
        tf.math.logical_or(sparsemax_values > 0, tf.math.is_nan(sparsemax_values)),
        sparsemax_values * (z - 0.5 * sparsemax_values),
        tf.zeros_like(sparsemax_values),
    )
    q_part = labels * (0.5 * labels - z)

    q_part_safe = tf.where(
        tf.math.logical_and(tf.math.equal(labels, 0), tf.math.is_inf(z)),
        tf.zeros_like(z),
        q_part,
    )

    loss = tf.math.reduce_sum(sum_s + q_part_safe, axis=1)

    return loss


# taken from Tensorflow-Addons (https://www.tensorflow.org/addons)
def _compute_2d_sparsemax(logits):
    """Performs the sparsemax operation when axis=-1."""
    shape_op = tf.shape(logits)
    obs = tf.math.reduce_prod(shape_op[:-1])
    dims = shape_op[-1]

    # In the paper, they call the logits z.
    # The mean(logits) can be substracted from logits to make the algorithm
    # more numerically stable. the instability in this algorithm comes mostly
    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
    # to zero. However, in practise the numerical instability issues are very
    # minor and substacting the mean causes extra issues with inf and nan
    # input.
    # Reshape to [obs, dims] as it is almost free and means the remanining
    # code doesn't need to worry about the rank.
    z = tf.reshape(logits, [obs, dims])

    # sort z
    z_sorted, _ = tf.nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
    k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)

    # calculate tau(z)
    # If there are inf values or all values are -inf, the k_z will be zero,
    # this is mathematically invalid and will also cause the gather_nd to fail.
    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
    # fixed later (see p_safe) by returning p = nan. This results in the same
    # behavior as softmax.
    k_z_safe = tf.math.maximum(k_z, 1)
    indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

    # calculate p
    p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
    # If k_z = 0 or if z = nan, then the input is invalid
    p_safe = tf.where(
        tf.expand_dims(
            tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
            axis=-1,
        ),
        tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
        p,
    )

    # Reshape back to original size
    p_safe = tf.reshape(p_safe, shape_op)
    return p_safe


# taken from Tensorflow-Addons (https://www.tensorflow.org/addons)
class SparsemaxLoss(tf.keras.losses.Loss):
    """Sparsemax loss function.

    Computes the generalized multi-label classification loss for the sparsemax
    function.

    Because the sparsemax loss function needs both the probability output and
    the logits to compute the loss value, `from_logits` must be `True`.

    Because it computes the generalized multi-label loss, the shape of both
    `y_pred` and `y_true` must be `[batch_size, num_classes]`.

    Args:
      from_logits: Whether `y_pred` is expected to be a logits tensor. Default
        is `True`, meaning `y_pred` is the logits.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `SUM_OVER_BATCH_SIZE`.
      name: Optional name for the op.
    """

    def __init__(
        self,
        from_logits: bool = True,
        reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "sparsemax_loss",
    ):
        if from_logits is not True:
            raise ValueError("from_logits must be True")

        super().__init__(name=name, reduction=reduction)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        return sparsemax_loss_from_logits(y_true, y_pred)

    def get_config(self):
        config = {
            "from_logits": self.from_logits,
        }
        base_config = super().get_config()
        return {**base_config, **config}
