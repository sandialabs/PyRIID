# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains custom loss functions."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


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

    sigma_term = tf.math.log(2 * np.pi * var)
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

    sigma_term = tf.math.log(2 * np.pi * sample_variance)

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
